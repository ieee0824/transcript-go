//go:build !arm64 && !amd64

package simd

// ButterflyBlock performs FFT butterfly operations on split real/imaginary arrays.
// For k in 0..len(uRe)-1:
//
//	t_re = twRe[k]*vRe[k] - twIm[k]*vIm[k]
//	t_im = twRe[k]*vIm[k] + twIm[k]*vRe[k]
//	uRe[k], vRe[k] = uRe[k]+t_re, uRe[k]-t_re
//	uIm[k], vIm[k] = uIm[k]+t_im, uIm[k]-t_im
func ButterflyBlock(uRe, uIm, vRe, vIm, twRe, twIm []float64) {
	for k := range uRe {
		tre := twRe[k]*vRe[k] - twIm[k]*vIm[k]
		tim := twRe[k]*vIm[k] + twIm[k]*vRe[k]
		ur := uRe[k]
		ui := uIm[k]
		uRe[k] = ur + tre
		uIm[k] = ui + tim
		vRe[k] = ur - tre
		vIm[k] = ui - tim
	}
}
