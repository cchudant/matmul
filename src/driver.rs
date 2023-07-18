use num::integer::div_ceil;
use std::mem::size_of;

use crate::kernel::{AVX2Kernel, LoopOrder, TileKernelArguments, TileKernelFlags};

#[derive(Debug, Clone)]
pub struct MatInfo {
    start: *mut u8,
    rows: isize,
    cols: isize,
    row_stride: isize,
    col_stride: isize,
}

impl MatInfo {
    fn is_contig_row(&self) -> bool {
        self.row_stride == size_of::<f32>() as _
            && self.start.align_offset(ALIGN_CONTIGUOUS_DIM_TO) == 0
            && (self.cols * self.col_stride) % ALIGN_CONTIGUOUS_DIM_TO as isize == 0
    }

    fn effective_size_cache_lines(&self) -> usize {
        div_ceil(
            self.rows as usize
                * self.row_stride.abs() as usize
                * 4
                * self.cols as usize
                * self.col_stride.abs() as usize
                * 4,
            ALIGN_CONTIGUOUS_DIM_TO,
        )
    }
}

const ALIGN_CONTIGUOUS_DIM_TO: usize = 32;

pub struct MacroKernelInfo {
    a: MatInfo,
    b: MatInfo,
    c: MatInfo,
}

#[derive(Clone, Copy, Debug)]
enum Matrix {
    A,
    B,
    C,
}

fn select_kernel_mix(
    _outer_dim: usize,
    middle_dim: usize,
    inner_dim: usize,
    loop_order: LoopOrder,
) -> KernelMix {
    use crate::extern_kernels::*;

    let n_elem_simd = 8;
    let main = (40, 2, loop_order);

    let border1 = (main.0, inner_dim % main.1, loop_order);
    let border2 = (
        div_ceil(middle_dim % main.0, n_elem_simd) * n_elem_simd,
        main.1,
        loop_order,
    );
    let border3 = (border2.0, border1.1, loop_order);

    println!("{:?} {:?} {:?} {:?}", main, border1, border2, border3);

    KernelMix {
        main: select_avx2_kernel(main.0, main.1, main.2).expect("unexpected missing kernel"),
        border1: select_avx2_kernel(border1.0, border1.1, border1.2),
        border2: select_avx2_kernel(border2.0, border2.1, border2.2),
        border3: select_avx2_kernel(border3.0, border3.1, border3.2),
    }
}

#[inline]
fn call_kernel(ker: &'static AVX2Kernel, args: &TileKernelArguments) {
    if cfg!(debug_assertions) {
        println!("Call {} with {:?}", ker, args);

        assert!(args.middle_lm <= ker.r as _);
        assert!(args.middle_lm > 0);
        assert!(args.outer_len > 0);

        assert!(args.inner_stride_0 >= size_of::<f32>() as _);
        assert!(args.inner_stride_1 >= size_of::<f32>() as _);
        assert!(args.middle_stride_0 >= size_of::<f32>() as _);
        assert!(args.middle_stride_1 >= size_of::<f32>() as _);
        assert!(args.outer_stride_0 >= size_of::<f32>() as _);
        assert!(args.outer_stride_1 >= size_of::<f32>() as _);

        if args.middle_lm < ker.r {
            assert!(!args.flags.contains(TileKernelFlags::MIDDLE_IS_CONTIG))
        }
        if args.flags.contains(TileKernelFlags::MIDDLE_IS_CONTIG) {
            assert_eq!(args.middle_stride_0, size_of::<f32>() as _);
            assert_eq!(args.middle_matrix.align_offset(ALIGN_CONTIGUOUS_DIM_TO), 0);
            assert_eq!(
                args.middle_matrix
                    .wrapping_offset(args.middle_stride_0 * ker.r as isize)
                    .align_offset(ALIGN_CONTIGUOUS_DIM_TO),
                0
            );
        }
        if args.flags.contains(TileKernelFlags::OUTER_IS_CONTIG) {
            assert_eq!(args.outer_stride_0, size_of::<f32>() as _);
            assert_eq!(args.outer_matrix.align_offset(ALIGN_CONTIGUOUS_DIM_TO), 0);
            assert_eq!(
                args.outer_matrix
                    .wrapping_offset(args.outer_stride_0 * ker.r as isize)
                    .align_offset(ALIGN_CONTIGUOUS_DIM_TO),
                0
            );
        }
    }

    let ret = unsafe { (ker.func)(args as _) };
    if cfg!(debug_assertions) {
        assert_eq!(ret, 0);
    }
}

#[derive(Debug)]
struct KernelMix {
    /// main kernel
    main: &'static AVX2Kernel,
    /// border on mid dimension
    border1: Option<&'static AVX2Kernel>,
    /// border on inner dimension
    border2: Option<&'static AVX2Kernel>,
    /// border on mid and inner dimension
    border3: Option<&'static AVX2Kernel>,
}

#[derive(Debug)]
pub struct DriverPlan {
    outer_matrix: MatInfo,
    middle_matrix: MatInfo,
    inner_matrix: MatInfo,
    kernels: KernelMix,
    outer_dim: usize,
    middle_dim: usize,
    inner_dim: usize,
    flags: TileKernelFlags,
}

pub fn make_plan(mut info: MacroKernelInfo) -> DriverPlan {
    assert!(info.a.rows > 1);
    assert!(info.a.cols > 1);
    assert!(info.b.rows > 1);
    assert!(info.b.cols > 1);
    assert!(info.c.rows > 1);
    assert!(info.c.cols > 1);

    let assert_no_zero = |mat: &MatInfo| {
        assert_ne!(mat.row_stride, 0);
        assert_ne!(mat.rows, 0);
        assert_ne!(mat.col_stride, 0);
        assert_ne!(mat.cols, 0);
    };

    assert_no_zero(&info.a);
    assert_no_zero(&info.b);
    assert_no_zero(&info.c);

    assert_eq!(info.a.rows, info.c.rows); // m
    assert_eq!(info.a.cols, info.b.rows); // k
    assert_eq!(info.b.cols, info.c.cols); // n

    // reverse the matrices so that strides are positive
    let rev_matrix_if_needed = |mat: &mut MatInfo| {
        if mat.row_stride < 0 {
            mat.start = unsafe { mat.start.offset(mat.rows * mat.row_stride) };
            mat.row_stride = -mat.row_stride;
        }
        if mat.col_stride < 0 {
            mat.start = unsafe { mat.start.offset(mat.cols * mat.col_stride) };
            mat.col_stride = -mat.col_stride;
        }
    };

    rev_matrix_if_needed(&mut info.a);
    rev_matrix_if_needed(&mut info.b);
    rev_matrix_if_needed(&mut info.c);

    let es_a = info.a.effective_size_cache_lines();
    let es_b = info.b.effective_size_cache_lines();
    let es_c = info.c.effective_size_cache_lines();

    // hopefully llvm figures out something less stupid here
    // it'll do for now..
    let outer_matrix = if es_b > es_a && es_c > es_a {
        (&info.a, Matrix::A)
    } else if es_a > es_b && es_c > es_b {
        (&info.b, Matrix::B)
    } else {
        (&info.c, Matrix::C)
    };
    let middle_matrix = if es_b > es_a && es_a > es_c {
        (&info.a, Matrix::A)
    } else if es_a > es_b && es_b > es_c {
        (&info.b, Matrix::B)
    } else {
        (&info.c, Matrix::C)
    };
    let inner_matrix = if es_a > es_b && es_a > es_c {
        (&info.a, Matrix::A)
    } else if es_b > es_a && es_b > es_c {
        (&info.b, Matrix::B)
    } else {
        (&info.c, Matrix::C)
    };

    let get_dim = |mat| match mat {
        Matrix::A => info.c.cols, // on n
        Matrix::B => info.a.rows, // on m
        Matrix::C => info.a.cols, // on k
    };
    let outer_dim = get_dim(outer_matrix.1);
    let middle_dim = get_dim(middle_matrix.1);
    let inner_dim = get_dim(inner_matrix.1);
    println!("DIMS {}, {}, {}", outer_dim, middle_dim, inner_dim);
    println!(
        "DIMS {:?}, {:?}, {:?}",
        outer_matrix, middle_matrix, inner_matrix
    );

    let loop_order = match (outer_matrix.1, middle_matrix.1, inner_matrix.1) {
        (Matrix::C, Matrix::A, Matrix::B) => LoopOrder::CAB,
        (Matrix::C, Matrix::B, Matrix::A) => LoopOrder::CBA,
        (Matrix::A, Matrix::C, Matrix::B) => LoopOrder::ACB,
        (Matrix::B, Matrix::C, Matrix::A) => LoopOrder::BCA,
        (Matrix::A, Matrix::B, Matrix::C) => LoopOrder::ABC,
        (Matrix::B, Matrix::A, Matrix::C) => LoopOrder::BAC,
        _ => unreachable!(),
    };

    let kernels = select_kernel_mix(outer_dim as _, middle_dim as _, inner_dim as _, loop_order);
    println!("{:?}", kernels);

    let mut flags = TileKernelFlags::OVERWRITE_C;
    if outer_matrix.0.is_contig_row() {
        flags |= TileKernelFlags::OUTER_IS_CONTIG;
    }
    if middle_matrix.0.is_contig_row() {
        flags |= TileKernelFlags::MIDDLE_IS_CONTIG;
    }

    DriverPlan {
        outer_matrix: outer_matrix.0.clone(),
        middle_matrix: middle_matrix.0.clone(),
        inner_matrix: inner_matrix.0.clone(),
        kernels,
        outer_dim: outer_dim as _,
        middle_dim: middle_dim as _,
        inner_dim: inner_dim as _,
        flags,
    }
}

pub unsafe fn kernel_driver(plan: &DriverPlan) {
    let DriverPlan {
        outer_matrix,
        middle_matrix,
        inner_matrix,
        kernels,
        outer_dim,
        middle_dim,
        inner_dim,
        flags,
    } = plan;

    let outer_matrix_ptr = outer_matrix.start;
    let middle_matrix_ptr = middle_matrix.start;
    let inner_matrix_ptr = inner_matrix.start;
    let mut args = TileKernelArguments {
        outer_len: *outer_dim,
        middle_lm: kernels.main.r,
        outer_matrix: outer_matrix_ptr,
        middle_matrix: middle_matrix_ptr,
        inner_matrix: inner_matrix_ptr,
        outer_stride_0: outer_matrix.row_stride * size_of::<f32>() as isize,
        outer_stride_1: outer_matrix.col_stride * size_of::<f32>() as isize,
        middle_stride_0: middle_matrix.row_stride * size_of::<f32>() as isize,
        middle_stride_1: middle_matrix.col_stride * size_of::<f32>() as isize,
        inner_stride_0: inner_matrix.row_stride * size_of::<f32>() as isize,
        inner_stride_1: inner_matrix.col_stride * size_of::<f32>() as isize,
        flags: *flags,
    };

    let mut outer_ptr_0 = outer_matrix_ptr;
    let mut mid_ptr_0 = middle_matrix_ptr;
    for i_mid in (0..*middle_dim).step_by(kernels.main.r) {
        let mut outer_ptr_1 = outer_ptr_0;
        let mut inner_ptr_1 = inner_matrix_ptr;

        let rem_elems_mid = middle_dim - i_mid;
        if rem_elems_mid >= kernels.main.r as _ {
            // we are not in the border on the inn dimension

            let main_kernel = kernels.main;
            for i_inner in (0..*inner_dim).step_by(kernels.main.s) {
                let rem_elem_inn = middle_dim - i_inner;
                let kernel = if rem_elem_inn >= kernels.main.s as _ {
                    main_kernel
                } else {
                    // we are on the border in the inner dimension only
                    kernels.border1.expect("unexpected missing kernel")
                };

                args.outer_matrix = outer_ptr_1;
                args.middle_matrix = mid_ptr_0;
                args.inner_matrix = inner_ptr_1;

                call_kernel(kernel, &args);

                outer_ptr_1 =
                    unsafe { outer_ptr_1.offset(args.outer_stride_0 * kernels.main.r as isize) };
                inner_ptr_1 =
                    unsafe { inner_ptr_1.offset(args.inner_stride_0 * kernels.main.s as isize) };
            }
        } else {
            // we are on the border on the mid dimension

            args.middle_lm = rem_elems_mid as _;
            args.flags.remove(TileKernelFlags::MIDDLE_IS_CONTIG);

            let ker = kernels.border2.expect("unexpected missing kernel");
            for i_inner in (0..*inner_dim).step_by(kernels.main.s) {
                let rem_elems_inn = inner_dim - i_inner;
                let kernel = if rem_elems_inn >= kernels.main.s as _ {
                    ker
                } else {
                    // we are on the border in the mid and inner dimension
                    kernels.border3.expect("unexpected missing kernel")
                };

                args.outer_matrix = outer_ptr_1;
                args.middle_matrix = mid_ptr_0;
                args.inner_matrix = inner_ptr_1;

                call_kernel(kernel, &args);

                outer_ptr_1 =
                    unsafe { outer_ptr_1.offset(args.outer_stride_0 * kernels.main.r as isize) };
                inner_ptr_1 =
                    unsafe { inner_ptr_1.offset(args.inner_stride_0 * kernels.main.s as isize) };
            }
        }

        println!("mid loop");

        outer_ptr_0 = unsafe { outer_ptr_0.offset(args.outer_stride_1 * kernels.main.s as isize) };
        mid_ptr_0 = unsafe { mid_ptr_0.offset(args.middle_stride_0 * kernels.main.r as isize) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::tests::make_matrix;

    #[test]
    fn test_ndarray() {
        let n = 30;
        let k = 10;
        let m = 3;

        let mut a = make_matrix::<f32>(k, m, false);
        let mut b = make_matrix::<f32>(n, k, false);
        let mut c = make_matrix::<f32>(n, m, true);

        println!(
            "A goes from {:?} {:?}",
            a.first().map(|e| e as *const f32),
            a.last().map(|e| e as *const f32)
        );
        println!(
            "B goes from {:?} {:?}",
            b.first().map(|e| e as *const f32),
            b.last().map(|e| e as *const f32)
        );
        println!(
            "C goes from {:?} {:?}",
            c.first().map(|e| e as *const f32),
            c.last().map(|e| e as *const f32)
        );

        let plan = make_plan(MacroKernelInfo {
            a: MatInfo {
                start: a.as_mut_ptr() as _,
                rows: a.shape()[0] as _,
                cols: a.shape()[1] as _,
                row_stride: a.strides()[1],
                col_stride: a.strides()[0],
            },
            b: MatInfo {
                start: b.as_mut_ptr() as _,
                rows: b.shape()[0] as _,
                cols: b.shape()[1] as _,
                row_stride: b.strides()[1],
                col_stride: b.strides()[0],
            },
            c: MatInfo {
                start: c.as_mut_ptr() as _,
                rows: c.shape()[0] as _,
                cols: c.shape()[1] as _,
                row_stride: c.strides()[1],
                col_stride: c.strides()[0],
            },
        });

        println!("PLAN: {:?}", plan);

        unsafe { kernel_driver(&plan) }

        println!("{:?}", c);

        println!("{:?}", a.dot(&b));
    }
}
