use half::{bf16, f16};
use ndarray::{Array, Array2, Dimension, Ix2, OwnedRepr};
use num::integer::div_ceil;
use std::{
    default,
    fmt::{Debug, Display},
    mem::{size_of, MaybeUninit},
    ops::{Add, Mul},
    ptr::{null, null_mut},
};

use crate::extern_kernels;

pub trait DType: Sized + Copy + Add + Mul + Default + Display + Debug {
    const SIZE_OF: usize = size_of::<Self>();
}

impl DType for f32 {}
impl DType for bf16 {}
impl DType for f16 {}

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum ContiguousDimension {
    A_M = 0,
    B_N = 1,
}

// #[repr(u8)]
// #[derive(Clone, Copy, PartialEq, Eq)]
// #[allow(non_camel_case_types)]
// pub enum LoopOrder {
//     ABC = 0, // n,m,k
//     BAC = 1, // m,n,k
//     ACB = 2, // n,k.m
//     BCA = 3, // m,k,n
//     CAB = 4, // k,n,m
//     CBA = 5, // k,m,n
// }

// / The microkernel must be called with at least one side of the input matrix that is:
// / - contiguous (stride = element size)
// / - properly aligned to the kernel prefered alignment
// / On top of that, the tile dimensions `lm`, `ln`, `lk` must be less than or equal to
// / the kernel prefered `M`, `N` and `K` tile dimensions respectively.
// /
// / Strides should not be nul or negative.
// /
// / The macrokernel (outer rust part) must use packing, padding and tiling to respect these conditions.
// /
// / The microkernel (asm part) must support:
// / - the different contiguous dimensions
// / - having the any dimension (including the continuous one) being less that the kernel prefered tile size
// /
// / The contiguous dimension can be less than the prefered tile size, because of padding added during packing
// / which should not count in the total result. The microkernel is expected to handle that using masking.
// /
// / If there is padding, it must be zeros.
// #[repr(C)]
// pub struct InnerKernelABI {
//     a_start: *const u8,
//     b_start: *const u8,
//     c_start: *mut u8,
//     contig_dim: isize,
//     tile_dim: isize,
//     lk: isize,
//     a_col_stride: isize, // m
//     a_row_stride: isize, // k
//     b_col_stride: isize, // k
//     b_row_stride: isize, // n
//     c_col_stride: isize, // m
//     c_row_stride: isize, // n
//     contiguous_dim: LoopOrder,
//     c_contig: bool,
// }

// impl InnerKernelABI {
//     fn assert_valid<K: GemmKernel>(&self) {
//         assert!(self.contig_dim <= K::CONTIG_DIM as _);
//         assert!(self.tile_dim <= K::TILE_DIM as _);

//         assert!(self.a_row_stride > 0);
//         assert!(self.a_col_stride > 0);
//         assert!(self.b_row_stride > 0);
//         assert!(self.b_col_stride > 0);
//         assert!(self.c_row_stride > 0);
//         assert!(self.c_col_stride > 0);

//         use ContiguousDimension::*;
//         match self.contiguous_dim {
//             A_M => {
//                 assert_eq!(self.a_col_stride, K::ATy::SIZE_OF as _);
//                 assert_eq!(self.a_start.align_offset(K::ALIGN_CONTIGUOUS_DIM_TO), 0);
//             }
//             B_N => {
//                 assert_eq!(self.b_col_stride, K::BTy::SIZE_OF as _);
//                 assert_eq!(self.b_start.align_offset(K::ALIGN_CONTIGUOUS_DIM_TO), 0);
//             }
//         }
//     }
// }

// / The packing kernel will only pad rows, not the columns.
// / You can call it with swapped dimensions/strides to pack the columns instead.
// #[repr(C)]
// pub struct PackKernelABI {
//     src: *const u8,
//     rows: isize,
//     row_stride: isize,
//     cols: isize,
//     col_stride: isize,
//     dest: *mut u8,
//     padding: isize,
// }

// pub trait GemmKernel {
//     type ATy: DType + Mul<Self::BTy, Output = Self::CTy>;
//     type BTy: DType;
//     type CTy: DType + Add<Output = Self::CTy>;

//     /// Prefered tile size for the contig dim
//     const CONTIG_DIM: usize;
//     /// Prefered for the other non-k dim
//     const TILE_DIM: usize;
//     /// Alignment requirement for the contiguous dimension.
//     const ALIGN_CONTIGUOUS_DIM_TO: usize;

//     unsafe fn execute(&self, info: *const InnerKernelABI);

//     unsafe fn pack_a(&self, info: *const PackKernelABI) {
//         naive_pack::<Self::ATy>(info)
//     }
//     unsafe fn pack_b(&self, info: *const PackKernelABI) {
//         naive_pack::<Self::BTy>(info)
//     }
// }

// pub struct NaiveKernel;
// impl GemmKernel for NaiveKernel {
//     const CONTIG_DIM: usize = 8;
//     const TILE_DIM: usize = 11;
//     const ALIGN_CONTIGUOUS_DIM_TO: usize = 32;
//     type ATy = f32;
//     type BTy = f32;
//     type CTy = f32;

//     unsafe fn execute(&self, info: *const InnerKernelABI) {
//         let info = info.read();
//         for i_m in 0..info.contig_dim as isize {
//             for i_n in 0..info.tile_dim as isize {
//                 for i_k in 0..info.lk as isize {
//                     let a_val = info
//                         .a_start
//                         .offset(i_m * info.a_col_stride + i_k)
//                         .cast::<Self::ATy>()
//                         .read();
//                     let b_val = info
//                         .b_start
//                         .offset(i_k * info.b_col_stride + i_n * info.b_row_stride)
//                         .cast::<Self::BTy>()
//                         .read();
//                     let c_ptr = info
//                         .c_start
//                         .offset(i_m * info.c_col_stride + i_n * info.c_row_stride)
//                         .cast::<Self::CTy>();

//                     let c_val = c_ptr.read();

//                     c_ptr.write(c_val + a_val * b_val)
//                 }
//             }
//         }
//     }
// }

// unsafe fn naive_pack<Ty: DType>(info: *const PackKernelABI) {
//     let mut info = info.read();

//     for j in 0..info.cols {
//         for i in 0..info.rows {
//             let val = info
//                 .src
//                 .offset(j * info.col_stride + i * info.row_stride)
//                 .cast::<Ty>()
//                 .read();
//             info.dest.cast::<Ty>().write(val);
//             info.dest = info.dest.add(Ty::SIZE_OF);
//         }

//         for i in 0..info.padding {
//             info.dest.cast::<Ty>().write(Default::default());
//             info.dest = info.dest.add(Ty::SIZE_OF);
//         }
//     }
// }

// pub struct MatInfo {
//     start: *mut u8,
//     rows: isize,
//     cols: isize,
//     row_stride: isize,
//     col_stride: isize,
// }

// pub struct MacroKernelInfo {
//     a: MatInfo,
//     b: MatInfo,
//     c: MatInfo,
// }

/*
/// safety:
/// - caller has to make sure the strides, rows and cols are valid for each matrices
/// so that we never end up out of bound
/// - the a and b matrices are initialized
/// - the c matrix can be written to (it may be uninitialized)
pub unsafe fn macrokernel<K: GemmKernel>(kernel: K, mut info: MacroKernelInfo) {
    use ContiguousDimension as Dim;

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

    let a_m_contig = info.a.row_stride == 1;
    let a_k_contig = info.a.col_stride == 1;
    let b_k_contig = info.b.row_stride == 1;
    let b_n_contig = info.b.col_stride == 1;

    let a_aligned_start = info.a.start.align_offset(K::ALIGN_CONTIGUOUS_DIM_TO) == 0;
    let b_aligned_start = info.b.start.align_offset(K::ALIGN_CONTIGUOUS_DIM_TO) == 0;

    let a_m_aligned = a_aligned_start
        && (info.a.cols * info.a.col_stride) % K::ALIGN_CONTIGUOUS_DIM_TO as isize == 0;
    let a_k_aligned = a_aligned_start
        && (info.a.rows * info.a.row_stride) % K::ALIGN_CONTIGUOUS_DIM_TO as isize == 0;
    let b_k_aligned = b_aligned_start
        && (info.a.cols * info.a.col_stride) % K::ALIGN_CONTIGUOUS_DIM_TO as isize == 0;
    let b_n_aligned = b_aligned_start
        && (info.a.rows * info.a.row_stride) % K::ALIGN_CONTIGUOUS_DIM_TO as isize == 0;

    let a_m_favorable = a_m_contig && a_m_aligned;
    let a_k_favorable = a_k_contig && a_k_aligned;
    let b_k_favorable = b_k_contig && b_k_aligned;
    let b_n_favorable = b_n_contig && b_n_aligned;

    let (do_packing, contig_dim) =
        match (a_m_favorable, a_k_favorable, b_n_favorable, b_k_favorable) {
            (true, _, _, _) => (false, Dim::A_M),
            (_, _, _, true) => (false, Dim::B_N),
            _ => (true, Dim::A_M),
        };

    // how do we tile this?
    // for now, dumb (we do no threading so its ok for now)

    let tiles_m = div_ceil(info.a.rows as usize, K::M);
    let tiles_n = div_ceil(info.b.cols as usize, K::N);
    let tiles_k = div_ceil(info.a.cols as usize, K::K);

    let stepper_m = (0..tiles_m)
        .map(|i| (i, i * K::M, (info.a.rows as usize - i * K::M).min(K::M)));
    let stepper_n = (0..tiles_n)
        .map(|i| (i, i * K::N, (info.b.cols as usize - i * K::N).min(K::N)));
    let stepper_k = (0..tiles_k)
        .map(|i| (i, i * K::K, (info.a.rows as usize - i * K::K).min(K::K)));

    let mut k_params = InnerKernelABI {
        a_start: null(),
        b_start: null(),
        c_start: null_mut(),
        contig_dim: 0,
        tile_dim: 0,
        lk: 0,
        a_col_stride: info.a.col_stride,
        a_row_stride: info.a.row_stride,
        b_col_stride: info.b.col_stride,
        b_row_stride: info.b.row_stride,
        c_col_stride: info.c.col_stride,
        c_row_stride: info.c.row_stride,
        contiguous_dim: contig_dim,
    };

    let mut pack_params = PackKernelABI {
        src: null(),
        rows: 0,
        row_stride: 0,
        cols: 0,
        col_stride: 0,
        dest: null_mut(),
        padding: 0,
    };

    let padding = K::M - info.a.rows as usize % K::M;

    let scratchpad: Box<[MaybeUninit<u8>]> = vec![MaybeUninit::uninit(); 1].into_boxed_slice();

    if do_packing {
        match contig_dim {
            Dim::A_M => {
                pack_params.row_stride = k_params.a_row_stride;
                pack_params.col_stride = k_params.a_col_stride;
                // pack_params.padding = K::M - info.a.cols / ;
                k_params.a_row_stride = K::ATy::SIZE_OF as _;
                k_params.a_col_stride = (K::M * K::ATy::SIZE_OF) as _;
            }
            Dim::A_K => {
                pack_params.row_stride = k_params.a_col_stride;
                pack_params.col_stride = k_params.a_row_stride;
                k_params.a_row_stride = K::ATy::SIZE_OF as _;
                k_params.a_col_stride = (K::K * K::ATy::SIZE_OF) as _;
            }
            Dim::B_K => {
                pack_params.row_stride = k_params.b_row_stride;
                pack_params.col_stride = k_params.b_col_stride;
                k_params.a_row_stride = K::BTy::SIZE_OF as _;
                k_params.a_col_stride = (K::K * K::BTy::SIZE_OF) as _;
            }
            Dim::B_N => {
                pack_params.row_stride = k_params.b_col_stride;
                pack_params.col_stride = k_params.b_row_stride;
                k_params.a_row_stride = K::BTy::SIZE_OF as _;
                k_params.a_col_stride = (K::N * K::BTy::SIZE_OF) as _;
            }
        }
    }

    for (tile_k, index_k, l_k) in stepper_k {
        for (tile_m, index_m, l_m) in stepper_m {
            for (tile_n, index_n, l_n) in stepper_n {
                k_params.a_start = unsafe {
                    info.a
                        .start
                        .offset(index_m as isize * k_params.a_row_stride + index_k as isize * k_params.a_col_stride)
                };
                k_params.b_start = unsafe {
                    info.b
                        .start
                        .offset(index_k as isize * k_params.b_row_stride + index_n as isize * k_params.b_col_stride)
                };
                k_params.c_start = unsafe {
                    info.c
                        .start
                        .offset(index_m as isize * k_params.c_row_stride + index_n as isize * k_params.c_col_stride)
                };

                k_params.contig_dim = l_m as _;
                k_params.tile_dim = l_n as _;
                k_params.lk = l_k as _;

                // packing
                if do_packing {
                    match contig_dim {
                        Dim::A_M => {
                            pack_params.src = k_params.a_start;
                            pack_params.rows = l_m as _;
                            pack_params.cols = l_k as _;
                        }
                        Dim::A_K => {
                            pack_params.src = k_params.a_start;
                            pack_params.rows = l_k as _;
                            pack_params.cols = l_m as _;
                        }
                        Dim::B_K => {
                            pack_params.src = k_params.b_start;
                            pack_params.rows = l_k as _;
                            pack_params.cols = l_n as _;
                        }
                        Dim::B_N => {
                            pack_params.src = k_params.b_start;
                            pack_params.rows = l_n as _;
                            pack_params.cols = l_k as _;
                        }
                    }

                    // run kernel

                    if contig_dim == Dim::A_M || contig_dim == Dim::A_K {
                        unsafe { kernel.pack_a(&pack_params as _) };
                    } else {
                        unsafe { kernel.pack_b(&pack_params as _) };
                    }

                    match contig_dim {
                        Dim::A_M | Dim::A_K => k_params.a_start = pack_params.dest as _,
                        Dim::B_K | Dim::B_N => k_params.b_start = pack_params.dest as _,
                    }
                }

                k_params.assert_valid::<K>();
                unsafe { kernel.execute(&k_params as _) };
            }
        }
    }
}

pub fn run_from_ndarray<K: GemmKernel>(
    kernel: K,
    a: Array2<K::ATy>,
    b: Array2<K::BTy>,
) -> Array2<K::CTy>
where
    K::ATy: Mul<K::BTy, Output = K::CTy>,
    K::CTy: Add<Output = K::CTy>,
{
    let m = a.shape()[0];
    let k = a.shape()[1];
    assert_eq!(k, b.shape()[0]);
    let n = b.shape()[1];
    let c = vec![Default::default(); 1024];
    println!("{c:?}");
    let mut c = unsafe { Array2::<K::CTy>::from_shape_vec_unchecked([m, n], c) };

    println!("{} {} {}", m, n, k);

    println!("{:?}", c.strides());

    let info = MacroKernelInfo {
        m,
        n,
        k,
        a_start: a.as_ptr(),
        b_start: b.as_ptr(),
        c_start: c.as_mut_ptr(),
        c_row_stride: c.strides()[1] as usize * K::CTy::SIZE_OF,
        c_col_stride: c.strides()[0] as usize * K::CTy::SIZE_OF,
    };

    unsafe {
        macrokernel(kernel, info);
    }

    println!("{:?}", c.clone().into_raw_vec());
    c
}

#[test]
pub fn test() {
    let a: ndarray::Array<f32, Ix2> = ndarray::arr2(&[[1f32, 2.], [3., 4.], [5., 6.]]);
    let b: ndarray::Array<f32, Ix2> = ndarray::arr2(&[[1f32, 2., 4.], [3., 4., 6.]]);

    dbg!(a.strides());
    dbg!(b.strides());

    let res = run_from_ndarray(TestKernel, a.clone(), b.clone());
    dbg!(&res);
    assert_eq!(&a.dot(&b), res)
}
 */
