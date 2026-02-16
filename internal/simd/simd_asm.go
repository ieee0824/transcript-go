//go:build arm64 || amd64

package simd

// MahalanobisAccum computes sum((x[i]-mean[i])^2 * invVar[i]) for i in 0..len(x)-1.
// Uses NEON on arm64, SSE2 on amd64.
func MahalanobisAccum(x, mean, invVar []float64) float64 {
	n := len(x)
	if n == 0 {
		return 0
	}
	return mahalanobisAsm(&x[0], &mean[0], &invVar[0], n)
}

//go:noescape
func mahalanobisAsm(x, mean, invVar *float64, n int) float64
