//go:build arm64 || amd64

package simd

// ButterflyBlock performs FFT butterfly operations on split real/imaginary arrays.
// Uses NEON on arm64, SSE2 on amd64.
func ButterflyBlock(uRe, uIm, vRe, vIm, twRe, twIm []float64) {
	n := len(uRe)
	if n == 0 {
		return
	}
	butterflyAsm(&uRe[0], &uIm[0], &vRe[0], &vIm[0], &twRe[0], &twIm[0], n)
}

//go:noescape
func butterflyAsm(uRe, uIm, vRe, vIm, twRe, twIm *float64, n int)
