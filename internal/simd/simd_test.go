package simd

import (
	"math"
	"testing"
)

// mahalanobisGo is the pure Go reference implementation.
func mahalanobisGo(x, mean, invVar []float64) float64 {
	maha := 0.0
	for i, xi := range x {
		diff := xi - mean[i]
		maha += diff * diff * invVar[i]
	}
	return maha
}

func TestMahalanobisAccum_basic(t *testing.T) {
	x := []float64{1, 2, 3, 4, 5}
	mean := []float64{0, 0, 0, 0, 0}
	invVar := []float64{1, 1, 1, 1, 1}

	got := MahalanobisAccum(x, mean, invVar)
	want := mahalanobisGo(x, mean, invVar) // 1+4+9+16+25 = 55

	if math.Abs(got-want) > 1e-10 {
		t.Errorf("MahalanobisAccum = %f, want %f", got, want)
	}
}

func TestMahalanobisAccum_39dim(t *testing.T) {
	x := make([]float64, 39)
	mean := make([]float64, 39)
	invVar := make([]float64, 39)

	for i := range x {
		x[i] = float64(i) * 0.1
		mean[i] = float64(i) * 0.05
		invVar[i] = 1.0 + float64(i)*0.01
	}

	got := MahalanobisAccum(x, mean, invVar)
	want := mahalanobisGo(x, mean, invVar)

	if math.Abs(got-want) > 1e-10 {
		t.Errorf("MahalanobisAccum(39d) = %f, want %f", got, want)
	}
}

func TestMahalanobisAccum_empty(t *testing.T) {
	got := MahalanobisAccum(nil, nil, nil)
	if got != 0 {
		t.Errorf("MahalanobisAccum(nil) = %f, want 0", got)
	}
}

func TestMahalanobisAccum_single(t *testing.T) {
	x := []float64{3.0}
	mean := []float64{1.0}
	invVar := []float64{2.0}

	got := MahalanobisAccum(x, mean, invVar)
	want := (3.0 - 1.0) * (3.0 - 1.0) * 2.0 // 8.0

	if math.Abs(got-want) > 1e-10 {
		t.Errorf("MahalanobisAccum(single) = %f, want %f", got, want)
	}
}

func TestMahalanobisAccum_even(t *testing.T) {
	x := []float64{1, 2, 3, 4}
	mean := []float64{0.5, 1.5, 2.5, 3.5}
	invVar := []float64{2.0, 3.0, 4.0, 5.0}

	got := MahalanobisAccum(x, mean, invVar)
	want := mahalanobisGo(x, mean, invVar)

	if math.Abs(got-want) > 1e-10 {
		t.Errorf("MahalanobisAccum(even) = %f, want %f", got, want)
	}
}

func TestMahalanobisAccum_odd(t *testing.T) {
	x := []float64{1, 2, 3}
	mean := []float64{0.5, 1.5, 2.5}
	invVar := []float64{2.0, 3.0, 4.0}

	got := MahalanobisAccum(x, mean, invVar)
	want := mahalanobisGo(x, mean, invVar)

	if math.Abs(got-want) > 1e-10 {
		t.Errorf("MahalanobisAccum(odd) = %f, want %f", got, want)
	}
}

func BenchmarkMahalanobisAccum_39dim(b *testing.B) {
	x := make([]float64, 39)
	mean := make([]float64, 39)
	invVar := make([]float64, 39)
	for i := range x {
		x[i] = float64(i) * 0.1
		mean[i] = float64(i) * 0.05
		invVar[i] = 1.0
	}
	b.ResetTimer()
	for b.Loop() {
		MahalanobisAccum(x, mean, invVar)
	}
}
