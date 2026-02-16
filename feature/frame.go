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

// Frame splits samples into overlapping frames.
// frameLen and frameShift are in number of samples.
func Frame(samples []float64, frameLen, frameShift int) [][]float64 {
	n := len(samples)
	if n < frameLen {
		return nil
	}
	numFrames := 1 + (n-frameLen)/frameShift
	frames := make([][]float64, numFrames)
	for i := 0; i < numFrames; i++ {
		start := i * frameShift
		frame := make([]float64, frameLen)
		copy(frame, samples[start:start+frameLen])
		frames[i] = frame
	}
	return frames
}

// HammingWindow applies a Hamming window in-place.
func HammingWindow(frame []float64) {
	n := len(frame)
	for i := range frame {
		frame[i] *= 0.54 - 0.46*math.Cos(2*math.Pi*float64(i)/float64(n-1))
	}
}
