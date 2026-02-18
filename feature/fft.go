package feature

import (
	"math"
	"math/cmplx"
)

// FFT computes the radix-2 Cooley-Tukey FFT.
// Input length must be a power of 2.
func FFT(x []complex128) []complex128 {
	n := len(x)
	if n <= 1 {
		out := make([]complex128, n)
		copy(out, x)
		return out
	}

	// Bit-reversal permutation
	bits := 0
	for v := n; v > 1; v >>= 1 {
		bits++
	}
	result := make([]complex128, n)
	for i := 0; i < n; i++ {
		result[bitReverse(i, bits)] = x[i]
	}

	// Butterfly operations
	for size := 2; size <= n; size *= 2 {
		halfSize := size / 2
		w := cmplx.Exp(complex(0, -2*math.Pi/float64(size)))
		for start := 0; start < n; start += size {
			wn := complex(1, 0)
			for k := 0; k < halfSize; k++ {
				u := result[start+k]
				t := wn * result[start+k+halfSize]
				result[start+k] = u + t
				result[start+k+halfSize] = u - t
				wn *= w
			}
		}
	}
	return result
}

func bitReverse(x, bits int) int {
	var result int
	for i := 0; i < bits; i++ {
		result = (result << 1) | (x & 1)
		x >>= 1
	}
	return result
}

// fftWorkspace holds reusable buffers for FFT computation.
type fftWorkspace struct {
	buf      []complex128   // [fftSize]
	power    []float64      // [fftSize/2+1]
	bits     int
	perm     []int            // bit-reversal permutation table
	twiddles [][]complex128   // twiddle factors per butterfly stage
}

func newFFTWorkspace(fftSize int) *fftWorkspace {
	bits := 0
	for v := fftSize; v > 1; v >>= 1 {
		bits++
	}

	// Pre-compute bit-reversal permutation
	perm := make([]int, fftSize)
	for i := 0; i < fftSize; i++ {
		perm[i] = bitReverse(i, bits)
	}

	// Pre-compute twiddle factors for each butterfly stage
	var twiddles [][]complex128
	for size := 2; size <= fftSize; size *= 2 {
		halfSize := size / 2
		tw := make([]complex128, halfSize)
		w := cmplx.Exp(complex(0, -2*math.Pi/float64(size)))
		wn := complex(1, 0)
		for k := 0; k < halfSize; k++ {
			tw[k] = wn
			wn *= w
		}
		twiddles = append(twiddles, tw)
	}

	return &fftWorkspace{
		buf:      make([]complex128, fftSize),
		power:    make([]float64, fftSize/2+1),
		bits:     bits,
		perm:     perm,
		twiddles: twiddles,
	}
}

// computePowerSpectrum loads frame into buffer with optional windowing,
// performs in-place FFT, and writes power spectrum into ws.power. No allocations.
// If window is non-nil, it is applied during loading (fused window+load).
func (ws *fftWorkspace) computePowerSpectrum(frame []float64, window []float64) {
	n := len(ws.buf)
	frameLen := len(frame)
	// Load frame with optional windowing, zero-pad
	if window != nil {
		for i := range ws.buf {
			if i < frameLen {
				ws.buf[i] = complex(frame[i]*window[i], 0)
			} else {
				ws.buf[i] = 0
			}
		}
	} else {
		for i := range ws.buf {
			if i < frameLen {
				ws.buf[i] = complex(frame[i], 0)
			} else {
				ws.buf[i] = 0
			}
		}
	}

	// In-place bit-reversal using pre-computed permutation
	for i := 0; i < n; i++ {
		j := ws.perm[i]
		if i < j {
			ws.buf[i], ws.buf[j] = ws.buf[j], ws.buf[i]
		}
	}

	// In-place butterfly with pre-computed twiddle factors
	for stage, size := 0, 2; size <= n; stage, size = stage+1, size*2 {
		halfSize := size / 2
		tw := ws.twiddles[stage]
		for start := 0; start < n; start += size {
			for k := 0; k < halfSize; k++ {
				u := ws.buf[start+k]
				t := tw[k] * ws.buf[start+k+halfSize]
				ws.buf[start+k] = u + t
				ws.buf[start+k+halfSize] = u - t
			}
		}
	}

	// Power spectrum
	nBins := n/2 + 1
	fn := float64(n)
	for i := 0; i < nBins; i++ {
		r := real(ws.buf[i])
		im := imag(ws.buf[i])
		ws.power[i] = (r*r + im*im) / fn
	}
}

// PowerSpectrum computes |FFT(x)|^2 / N for a real-valued frame.
// The frame is zero-padded to fftSize.
// Returns the first fftSize/2+1 bins (positive frequencies).
func PowerSpectrum(frame []float64, fftSize int) []float64 {
	x := make([]complex128, fftSize)
	for i := 0; i < len(frame) && i < fftSize; i++ {
		x[i] = complex(frame[i], 0)
	}

	X := FFT(x)

	nBins := fftSize/2 + 1
	power := make([]float64, nBins)
	fn := float64(fftSize)
	for i := 0; i < nBins; i++ {
		r := real(X[i])
		im := imag(X[i])
		power[i] = (r*r + im*im) / fn
	}
	return power
}
