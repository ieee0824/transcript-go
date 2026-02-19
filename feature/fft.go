package feature

import (
	"math"
	"math/cmplx"

	"github.com/ieee0824/transcript-go/internal/simd"
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
// Uses split real/imaginary layout for SIMD-friendly butterfly operations.
type fftWorkspace struct {
	bufRe []float64   // [fftSize] real part
	bufIm []float64   // [fftSize] imaginary part
	power []float64   // [fftSize/2+1]
	bits  int
	perm  []int         // bit-reversal permutation table
	twRe  [][]float64   // twiddle factors real parts per stage
	twIm  [][]float64   // twiddle factors imag parts per stage
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

	// Pre-compute twiddle factors in split R/I layout
	var twRe, twIm [][]float64
	for size := 2; size <= fftSize; size *= 2 {
		halfSize := size / 2
		re := make([]float64, halfSize)
		im := make([]float64, halfSize)
		w := cmplx.Exp(complex(0, -2*math.Pi/float64(size)))
		wn := complex(1, 0)
		for k := 0; k < halfSize; k++ {
			re[k] = real(wn)
			im[k] = imag(wn)
			wn *= w
		}
		twRe = append(twRe, re)
		twIm = append(twIm, im)
	}

	return &fftWorkspace{
		bufRe: make([]float64, fftSize),
		bufIm: make([]float64, fftSize),
		power: make([]float64, fftSize/2+1),
		bits:  bits,
		perm:  perm,
		twRe:  twRe,
		twIm:  twIm,
	}
}

// computePowerSpectrum loads frame into buffer with optional windowing,
// performs in-place FFT, and writes power spectrum into ws.power. No allocations.
// If window is non-nil, it is applied during loading (fused window+load).
func (ws *fftWorkspace) computePowerSpectrum(frame []float64, window []float64) {
	n := len(ws.bufRe)
	frameLen := len(frame)

	// Load real part with optional windowing, zero-pad; zero imaginary part
	if window != nil {
		for i := 0; i < frameLen; i++ {
			ws.bufRe[i] = frame[i] * window[i]
		}
	} else {
		copy(ws.bufRe[:frameLen], frame)
	}
	for i := frameLen; i < n; i++ {
		ws.bufRe[i] = 0
	}
	clear(ws.bufIm)

	// In-place bit-reversal using pre-computed permutation
	for i := 0; i < n; i++ {
		j := ws.perm[i]
		if i < j {
			ws.bufRe[i], ws.bufRe[j] = ws.bufRe[j], ws.bufRe[i]
			ws.bufIm[i], ws.bufIm[j] = ws.bufIm[j], ws.bufIm[i]
		}
	}

	// In-place butterfly with SIMD-accelerated block operations
	for stage, size := 0, 2; size <= n; stage, size = stage+1, size*2 {
		halfSize := size / 2
		for start := 0; start < n; start += size {
			simd.ButterflyBlock(
				ws.bufRe[start:start+halfSize],
				ws.bufIm[start:start+halfSize],
				ws.bufRe[start+halfSize:start+size],
				ws.bufIm[start+halfSize:start+size],
				ws.twRe[stage],
				ws.twIm[stage])
		}
	}

	// Power spectrum
	nBins := n/2 + 1
	fn := float64(n)
	for i := 0; i < nBins; i++ {
		r := ws.bufRe[i]
		im := ws.bufIm[i]
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
