//go:build darwin && cgo

package blas

/*
#cgo CFLAGS: -DACCELERATE_NEW_LAPACK
#cgo LDFLAGS: -framework Accelerate
#include <Accelerate/Accelerate.h>
*/
import "C"
import "unsafe"

// Dgemm performs C = alpha*op(A)*op(B) + beta*C using Apple Accelerate (AMX).
// All matrices are row-major. op(X) = X if trans=false, X^T if trans=true.
// A is (m x k) or (k x m) if transA, B is (k x n) or (n x k) if transB, C is (m x n).
func Dgemm(transA, transB bool, m, n, k int,
	alpha float64, a []float64, lda int,
	b []float64, ldb int,
	beta float64, c []float64, ldc int) {

	var ta, tb C.enum_CBLAS_TRANSPOSE
	if transA {
		ta = C.CblasTrans
	} else {
		ta = C.CblasNoTrans
	}
	if transB {
		tb = C.CblasTrans
	} else {
		tb = C.CblasNoTrans
	}

	C.cblas_dgemm(C.CblasRowMajor, ta, tb,
		C.int(m), C.int(n), C.int(k),
		C.double(alpha),
		(*C.double)(unsafe.Pointer(&a[0])), C.int(lda),
		(*C.double)(unsafe.Pointer(&b[0])), C.int(ldb),
		C.double(beta),
		(*C.double)(unsafe.Pointer(&c[0])), C.int(ldc))
}

// HasAccelerate returns true when Apple Accelerate framework is available.
func HasAccelerate() bool { return true }
