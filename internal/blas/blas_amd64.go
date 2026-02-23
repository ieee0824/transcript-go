//go:build amd64 && (!darwin || !cgo)

package blas

import "unsafe"

// Dgemm performs C = alpha*op(A)*op(B) + beta*C.
// Uses AVX2+FMA for the common transA=false, transB=true path.
func Dgemm(transA, transB bool, m, n, k int,
	alpha float64, a []float64, lda int,
	b []float64, ldb int,
	beta float64, c []float64, ldc int) {

	if !transA && transB {
		for i := 0; i < m; i++ {
			aPtr := (*float64)(unsafe.Pointer(&a[i*lda]))
			for j := 0; j < n; j++ {
				bPtr := (*float64)(unsafe.Pointer(&b[j*ldb]))
				dot := dotAVX2(aPtr, bPtr, k)
				idx := i*ldc + j
				if beta == 0 {
					c[idx] = alpha * dot
				} else {
					c[idx] = alpha*dot + beta*c[idx]
				}
			}
		}
		return
	}

	dgemmGeneric(transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}

func dgemmGeneric(transA, transB bool, m, n, k int,
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
