use ndarray::Ix2;
// use test_attn::gemm::{run_from_ndarray, TestKernel};

fn main() {
    let a: ndarray::Array<f32, Ix2> = ndarray::arr2(&[[1f32, 2.], [3., 4.], [5., 6.]]);
    let b: ndarray::Array<f32, Ix2> = ndarray::arr2(&[[1f32, 2., 4.], [3., 4., 6.]]);

    dbg!(a.strides());
    dbg!(b.strides());

    // let res = run_from_ndarray(TestKernel, a.clone(), b.clone());
    // dbg!(&res);
    // assert_eq!(&a.dot(&b), res)
}