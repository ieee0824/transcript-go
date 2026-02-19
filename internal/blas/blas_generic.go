//go:build !darwin || !cgo

package blas

// Dgemm performs C = alpha*op(A)*op(B) + beta*C in pure Go.
// All matrices are row-major. op(X) = X if trans=false, X^T if trans=true.
func Dgemm(transA, transB bool, m, n, k int,
	alpha float64, a []float64, lda int,
	b []float64, ldb int,
	beta float64, c []float64, ldc int) {

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := 0.0
			for p := 0; p < k; p++ {
				var aVal, bVal float64
				if transA {
					aVal = a[p*lda+i]
				} else {
					aVal = a[i*lda+p]
				}
				if transB {
					bVal = b[j*ldb+p]
				} else {
					bVal = b[p*ldb+j]
				}
				sum += aVal * bVal
			}
			c[i*ldc+j] = alpha*sum + beta*c[i*ldc+j]
		}
	}
}

// HasAccelerate returns false on non-darwin platforms.
func HasAccelerate() bool { return false }
