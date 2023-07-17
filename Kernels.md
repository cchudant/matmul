

Kernel has tile dimensions in CxTxO where C > T > O.

Compute effective matrix sizes for A, B, C.
Sort them in order.

6 choices:

## Group 1: CAB, CBA

CBA is CAB with A and B swapped. If kernel ATy = BTy, we can just use the same code.

CAB: loop order is k,n,m
```
/* c_ptr_0 = c_ptr */

Zero C (CxT elems)

/* a_ptr_0 := a_ptr */
For k in 0..lo:
  Load A (C elems) /* strided load a_ptr_0: middle_stride_0 */
  /* a_ptr_0 += C * middle_stride_0 */

  /* b_ptr_0 := b_ptr */
  For n 0..lt:
    Load B (1 elem) /* [b_ptr_0] bc */
    /* b_ptr_0 += inner_stride_0 */

    For m 0..lc:
      C[m][n] += A[m] + B
    EndFor
  EndFor
  /* b_ptr += inner_stride_1 */
  /* a_ptr += middle_stride_1 */
EndFor
StoreAcc C (CxT elems) /* strided update: outer_stride_0, outer_stride_1 */
```

C > A > B, which means it is good when k < n < m. This kernel handles outer-product like problems:
```
O O  X  O O O  =  O O O
O O     O O O     O O O
O O               O O O
O O               O O O
O O               O O O
```

## Group 2: ACB, BCA

Same remark with ACB vs BCA

ACB: loop order is n,k,m
```
Load A (CxT elems)
For n in 0..lo:
  Zero C (C elems)
  For k 0..lt:
    Load B (1 elem)
    For m 0..lc:
      C[m] += A[m][k] + B
    EndFor
  EndFor
  StoreAcc C (C elems)
EndFor
```

A > C > B, which means it is good when n < k < m.
```
O O O  X  O O  =  O O
O O O     O O     O O
O O O     O O     O O
O O O             O O
O O O             O O
```

## Group 3: ABC, BAC

Same remark with ABC vs BAC

ABC: loop order is n,m,k
```
Load A (CxT elems)
For n in 0..lo:
  Load B (C elems)
  For m 0..lt:
    Zero C (1 elem)
    For k 0..lc:
      C += A[m][k] + B[n]
    EndFor
    StoreAcc C (1 elem)
  EndFor
EndFor
```

A > B > C, which means it is good when n < m < k. This kernel handles inner-product like problems:
```
O O O O O  X  O O  =  O O
O O O O O     O O     O O
O O O O O     O O     O O
              O O
              O O
```



