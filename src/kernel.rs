#[cfg(test)]
use proptest::prelude::*;
use std::fmt::{self, Display};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(test, derive(proptest_derive::Arbitrary))]
#[allow(non_camel_case_types)]
pub enum LoopOrder {
    ABC, // n,m,k
    BAC, // m,n,k
    ACB, // n,k.m
    BCA, // m,k,n
    CAB, // k,n,m
    CBA, // k,m,n
}

impl Display for LoopOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                LoopOrder::ABC => "abc",
                LoopOrder::BAC => "bac",
                LoopOrder::ACB => "acb",
                LoopOrder::BCA => "bca",
                LoopOrder::CAB => "cab",
                LoopOrder::CBA => "cba",
            }
        )
    }
}

bitflags::bitflags! {
    #[derive(Clone, Copy, Default, Debug, PartialEq, Eq, Hash)]
    #[repr(C)]
    pub struct TileKernelFlags: u8 {
        /// Here, contig means contiguous on the first dimension AND simd aligned AND properly initialized.
        /// The other dimension can be strided.
        const OUTER_IS_CONTIG = 0b01;
        const MIDDLE_IS_CONTIG = 0b10;
        /// When set, this tells the kernel to do C := A*B instead of C += A*B.
        /// This flag only makes sense when the C matrix is the outer one.
        const OVERWRITE_C = 0b100;
    }
}

#[cfg(test)]
impl Arbitrary for TileKernelFlags {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;
    fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
        any::<(bool, bool, bool)>()
            .prop_map(|(a, b, c)| {
                let mut flags = Self::default();
                if a {
                    flags |= Self::OUTER_IS_CONTIG
                }
                if b {
                    flags |= Self::MIDDLE_IS_CONTIG
                }
                if c {
                    flags |= Self::OVERWRITE_C
                }
                flags
            })
            .boxed()
    }
}

#[derive(Clone, Debug)]
#[repr(C)]
pub(crate) struct TileKernelArguments {
    /// How many loops the outer loop should do
    pub outer_len: usize,
    /// How many elements in the tile on the middle (SIMD) direction.
    /// This is used for tiles that are smaller than the simd size.
    pub middle_lm: usize,

    pub outer_matrix: *mut u8,
    pub middle_matrix: *mut u8,
    pub inner_matrix: *mut u8,

    // strides are in number of bytes
    pub outer_stride_0: isize,
    pub outer_stride_1: isize,
    pub middle_stride_0: isize,
    pub middle_stride_1: isize,
    pub inner_stride_0: isize,
    pub inner_stride_1: isize,

    pub flags: TileKernelFlags,
}

#[derive(Debug)]
pub struct AVX2Kernel {
    pub(crate) func: unsafe extern "C" fn(arg: *const TileKernelArguments) -> isize,
    /// SIMD dimension
    pub(crate) r: usize,
    /// second tiling dimension
    pub(crate) s: usize,
    pub(crate) loop_order: LoopOrder,
}

impl Display for AVX2Kernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "mmkernel_avx2_sss_{}x{}_{}",
            self.r, self.s, self.loop_order
        )
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use ndarray::{s, Array2, IntoDimension, ShapeBuilder};
    use num::{traits::AsPrimitive, Zero};
    use proptest::sample;
    use std::mem::size_of;

    use super::*;
    use std::{
        alloc::{GlobalAlloc, Layout, System},
        mem::MaybeUninit,
    };

    pub(crate) fn make_matrix<T: Copy + 'static>(d0: usize, d1: usize, zero: bool) -> Array2<T>
    where
        usize: AsPrimitive<T>,
        T: Zero,
    {
        let len = d0 * d1;
        unsafe {
            let layout = Layout::from_size_align(len * size_of::<T>(), 64).unwrap();
            let ptr = System.alloc(layout);
            assert!(!ptr.is_null());

            let slice = std::slice::from_raw_parts_mut(ptr.cast::<MaybeUninit<T>>(), len);

            for (i, el) in slice.iter_mut().enumerate() {
                if zero {
                    el.write(T::zero());
                } else {
                    el.write((i + 1).as_());
                }
            }

            let vec = Vec::from_raw_parts(slice.as_mut_ptr() as _, len, len);

            let shape = [d1, d0].into_shape().strides([d0, 1].into_dimension());
            Array2::from_shape_vec(shape, vec).unwrap()
        }
    }

    fn run_kernel(
        kernel: &AVX2Kernel,
        flags: TileKernelFlags,
        outer_dim: usize,
        mut middle_lm: usize,
    ) {
        let order = kernel.loop_order;
        let mid_dim = kernel.r;
        let inner_dim = kernel.s;

        if flags.contains(TileKernelFlags::OUTER_IS_CONTIG)
            || flags.contains(TileKernelFlags::MIDDLE_IS_CONTIG)
        {
            middle_lm = mid_dim;
        }
        if middle_lm > mid_dim {
            middle_lm = mid_dim;
        }

        let is_overwrite = flags.contains(TileKernelFlags::OVERWRITE_C);

        let mut outer_matrix = make_matrix::<f32>(
            mid_dim,
            inner_dim,
            !is_overwrite && matches!(order, LoopOrder::CAB | LoopOrder::CBA),
        );
        let mut mid_matrix = make_matrix::<f32>(
            mid_dim,
            outer_dim,
            matches!(order, LoopOrder::ACB | LoopOrder::BCA),
        );
        let mut inner_matrix = make_matrix::<f32>(
            inner_dim,
            outer_dim,
            matches!(order, LoopOrder::ABC | LoopOrder::BAC),
        );

        println!("{:?}", outer_matrix);
        println!("{:?}", mid_matrix);
        println!("{:?}", inner_matrix);

        let args = TileKernelArguments {
            outer_len: outer_dim,
            middle_lm,
            outer_matrix: outer_matrix.as_mut_ptr() as _,
            middle_matrix: mid_matrix.as_mut_ptr() as _,
            inner_matrix: inner_matrix.as_mut_ptr() as _,
            outer_stride_0: outer_matrix.strides()[1] * size_of::<f32>() as isize,
            outer_stride_1: outer_matrix.strides()[0] * size_of::<f32>() as isize,
            middle_stride_0: mid_matrix.strides()[1] * size_of::<f32>() as isize,
            middle_stride_1: mid_matrix.strides()[0] * size_of::<f32>() as isize,
            inner_stride_0: inner_matrix.strides()[1] * size_of::<f32>() as isize,
            inner_stride_1: inner_matrix.strides()[0] * size_of::<f32>() as isize,
            flags,
        };
        println!("{:?}", args);

        let ret = unsafe { (kernel.func)(&args as _) };
        assert_eq!(ret, 0);

        println!("{:?}", outer_matrix.slice(s![.., ..middle_lm]));
        println!("{:?}\n\n", mid_matrix.t().slice(s![..middle_lm, ..]));

        let (truth, transpose, our_result) = match order {
            LoopOrder::ABC => (
                outer_matrix
                    .slice(s![.., ..middle_lm])
                    .dot(&mid_matrix.t().slice(s![..middle_lm, ..])),
                true,
                inner_matrix.view(),
            ),
            LoopOrder::BAC => todo!(),
            LoopOrder::ACB => (
                outer_matrix
                    .t()
                    .slice(s![..middle_lm, ..])
                    .dot(&inner_matrix.t()),
                true,
                mid_matrix.slice(s![.., ..middle_lm]),
            ),
            LoopOrder::BCA => todo!(),
            LoopOrder::CAB => (
                mid_matrix.t().slice(s![..middle_lm, ..]).dot(&inner_matrix),
                true,
                outer_matrix.slice(s![.., ..middle_lm]),
            ),
            LoopOrder::CBA => todo!(),
        };

        println!("MINE:{:?}\n\n", our_result);
        let proper = if transpose { truth.t() } else { truth.view() };
        println!("PROPER:{:?}\n\n", proper);
        assert_eq!(our_result, proper);
    }

    fn kernel_strategy() -> impl Strategy<Value = &'static AVX2Kernel> {
        sample::select(crate::extern_kernels::KERNELS.iter().collect::<Vec<_>>())
    }

    #[test]
    fn test() {
        use crate::extern_kernels::*;
        run_kernel(&MMKERNEL_AVX2_SSS_8X1_ABC, TileKernelFlags::empty(), 2, 7)
    }

    proptest! {
        #[test]
        fn test_add(kernel in kernel_strategy(), flags: TileKernelFlags, outer_dim in 1..100usize, middle_lm in 1..100usize) {
            run_kernel(kernel, flags, outer_dim, middle_lm)
        }
    }
}
