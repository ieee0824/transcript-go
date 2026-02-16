//go:build !arm64 && !amd64

package simd

// MahalanobisAccum computes sum((x[i]-mean[i])^2 * invVar[i]) for i in 0..len(x)-1.
// Pure Go fallback for architectures without SIMD assembly.
func MahalanobisAccum(x, mean, invVar []float64) float64 {
	maha := 0.0
	for i, xi := range x {
		diff := xi - mean[i]
		maha += diff * diff * invVar[i]
	}
	return maha
}
