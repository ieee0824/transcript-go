package feature

import (
	"math"
	"testing"
)

func generateSine(n int, freq float64) []float64 {
	samples := make([]float64, n)
	for i := range samples {
		samples[i] = math.Sin(2 * math.Pi * freq * float64(i) / 16000)
	}
	return samples
}

func BenchmarkFFT_512(b *testing.B) {
	x := make([]complex128, 512)
	for i := range x {
		x[i] = complex(math.Sin(2*math.Pi*float64(i)/512), 0)
	}
	b.ResetTimer()
	for b.Loop() {
		FFT(x)
	}
}

func BenchmarkPowerSpectrum_512(b *testing.B) {
	frame := make([]float64, 400)
	for i := range frame {
		frame[i] = math.Sin(2 * math.Pi * 440 * float64(i) / 16000)
	}
	b.ResetTimer()
	for b.Loop() {
		PowerSpectrum(frame, 512)
	}
}

func BenchmarkMelFilterbank_Apply(b *testing.B) {
	fb := NewMelFilterbank(26, 512, 16000, 0, 8000)
	ps := make([]float64, 257)
	for i := range ps {
		ps[i] = 0.01
	}
	b.ResetTimer()
	for b.Loop() {
		fb.Apply(ps)
	}
}

func BenchmarkExtract_1sec(b *testing.B) {
	samples := generateSine(16000, 440)
	cfg := DefaultConfig()
	b.ResetTimer()
	for b.Loop() {
		Extract(samples, cfg)
	}
}

func BenchmarkExtract_5sec(b *testing.B) {
	samples := generateSine(80000, 440)
	cfg := DefaultConfig()
	b.ResetTimer()
	for b.Loop() {
		Extract(samples, cfg)
	}
}

func BenchmarkExtract_30sec(b *testing.B) {
	samples := generateSine(480000, 440)
	cfg := DefaultConfig()
	b.ResetTimer()
	for b.Loop() {
		Extract(samples, cfg)
	}
}
