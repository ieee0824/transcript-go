package feature

import "math"

// PreEmphasize applies a first-order high-pass filter: y[n] = x[n] - alpha*x[n-1].
func PreEmphasize(samples []float64, alpha float64) []float64 {
	out := make([]float64, len(samples))
	out[0] = samples[0]
	for i := 1; i < len(samples); i++ {
		out[i] = samples[i] - alpha*samples[i-1]
	}
	return out
}

// Frame splits samples into overlapping frames as slice views (no copy).
// frameLen and frameShift are in number of samples.
// The returned slices share underlying memory with samples, so the caller
// must not modify them in-place if overlapping frames are used.
func Frame(samples []float64, frameLen, frameShift int) [][]float64 {
	n := len(samples)
	if n < frameLen {
		return nil
	}
	numFrames := 1 + (n-frameLen)/frameShift
	frames := make([][]float64, numFrames)
	for i := 0; i < numFrames; i++ {
		start := i * frameShift
		frames[i] = samples[start : start+frameLen]
	}
	return frames
}

// hammingTable caches pre-computed Hamming window coefficients by frame length.
var hammingTable = make(map[int][]float64)

// getHammingWindow returns pre-computed Hamming window coefficients for the given length.
func getHammingWindow(n int) []float64 {
	if w, ok := hammingTable[n]; ok {
		return w
	}
	w := make([]float64, n)
	invN := 1.0 / float64(n-1)
	for i := range w {
		w[i] = 0.54 - 0.46*math.Cos(2*math.Pi*float64(i)*invN)
	}
	hammingTable[n] = w
	return w
}

// HammingWindow applies a Hamming window in-place.
func HammingWindow(frame []float64) {
	w := getHammingWindow(len(frame))
	for i := range frame {
		frame[i] *= w[i]
	}
}
