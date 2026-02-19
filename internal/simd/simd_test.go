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

// butterflyGo is the pure Go reference implementation.
func butterflyGo(uRe, uIm, vRe, vIm, twRe, twIm []float64) {
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

func TestButterflyBlock_basic(t *testing.T) {
	// Simple butterfly: tw = 1+0i, so t = v, u'=u+v, v'=u-v
	uRe := []float64{1, 2, 3, 4}
	uIm := []float64{0, 0, 0, 0}
	vRe := []float64{5, 6, 7, 8}
	vIm := []float64{0, 0, 0, 0}
	twRe := []float64{1, 1, 1, 1}
	twIm := []float64{0, 0, 0, 0}

	ButterflyBlock(uRe, uIm, vRe, vIm, twRe, twIm)

	wantURe := []float64{6, 8, 10, 12}
	wantVRe := []float64{-4, -4, -4, -4}
	for i := range uRe {
		if math.Abs(uRe[i]-wantURe[i]) > 1e-10 {
			t.Errorf("uRe[%d] = %f, want %f", i, uRe[i], wantURe[i])
		}
		if math.Abs(vRe[i]-wantVRe[i]) > 1e-10 {
			t.Errorf("vRe[%d] = %f, want %f", i, vRe[i], wantVRe[i])
		}
	}
}

func TestButterflyBlock_complex(t *testing.T) {
	// Butterfly with complex twiddle factors
	n := 8
	uRe := make([]float64, n)
	uIm := make([]float64, n)
	vRe := make([]float64, n)
	vIm := make([]float64, n)
	twRe := make([]float64, n)
	twIm := make([]float64, n)

	for i := 0; i < n; i++ {
		uRe[i] = float64(i) * 0.3
		uIm[i] = float64(i) * 0.1
		vRe[i] = float64(i)*0.2 + 1
		vIm[i] = float64(i)*0.4 - 0.5
		theta := float64(i) * 0.7
		twRe[i] = math.Cos(theta)
		twIm[i] = math.Sin(theta)
	}

	// Reference
	wantURe := make([]float64, n)
	wantUIm := make([]float64, n)
	wantVRe := make([]float64, n)
	wantVIm := make([]float64, n)
	copy(wantURe, uRe)
	copy(wantUIm, uIm)
	copy(wantVRe, vRe)
	copy(wantVIm, vIm)
	butterflyGo(wantURe, wantUIm, wantVRe, wantVIm, twRe, twIm)

	ButterflyBlock(uRe, uIm, vRe, vIm, twRe, twIm)

	for i := 0; i < n; i++ {
		if math.Abs(uRe[i]-wantURe[i]) > 1e-10 {
			t.Errorf("uRe[%d] = %f, want %f", i, uRe[i], wantURe[i])
		}
		if math.Abs(uIm[i]-wantUIm[i]) > 1e-10 {
			t.Errorf("uIm[%d] = %f, want %f", i, uIm[i], wantUIm[i])
		}
		if math.Abs(vRe[i]-wantVRe[i]) > 1e-10 {
			t.Errorf("vRe[%d] = %f, want %f", i, vRe[i], wantVRe[i])
		}
		if math.Abs(vIm[i]-wantVIm[i]) > 1e-10 {
			t.Errorf("vIm[%d] = %f, want %f", i, vIm[i], wantVIm[i])
		}
	}
}

func TestButterflyBlock_odd(t *testing.T) {
	// Odd length: test tail handling (n=5 → 2 SIMD iterations + 1 scalar)
	n := 5
	uRe := make([]float64, n)
	uIm := make([]float64, n)
	vRe := make([]float64, n)
	vIm := make([]float64, n)
	twRe := make([]float64, n)
	twIm := make([]float64, n)

	for i := 0; i < n; i++ {
		uRe[i] = float64(i+1) * 1.1
		uIm[i] = float64(i+1) * 0.5
		vRe[i] = float64(i+1) * 0.7
		vIm[i] = float64(i+1) * 0.3
		theta := float64(i) * math.Pi / float64(n)
		twRe[i] = math.Cos(theta)
		twIm[i] = math.Sin(theta)
	}

	wantURe := make([]float64, n)
	wantUIm := make([]float64, n)
	wantVRe := make([]float64, n)
	wantVIm := make([]float64, n)
	copy(wantURe, uRe)
	copy(wantUIm, uIm)
	copy(wantVRe, vRe)
	copy(wantVIm, vIm)
	butterflyGo(wantURe, wantUIm, wantVRe, wantVIm, twRe, twIm)

	ButterflyBlock(uRe, uIm, vRe, vIm, twRe, twIm)

	for i := 0; i < n; i++ {
		if math.Abs(uRe[i]-wantURe[i]) > 1e-10 {
			t.Errorf("uRe[%d] = %f, want %f", i, uRe[i], wantURe[i])
		}
		if math.Abs(uIm[i]-wantUIm[i]) > 1e-10 {
			t.Errorf("uIm[%d] = %f, want %f", i, uIm[i], wantUIm[i])
		}
		if math.Abs(vRe[i]-wantVRe[i]) > 1e-10 {
			t.Errorf("vRe[%d] = %f, want %f", i, vRe[i], wantVRe[i])
		}
		if math.Abs(vIm[i]-wantVIm[i]) > 1e-10 {
			t.Errorf("vIm[%d] = %f, want %f", i, vIm[i], wantVIm[i])
		}
	}
}

func TestButterflyBlock_single(t *testing.T) {
	// Single element: only tail path
	uRe := []float64{3.0}
	uIm := []float64{4.0}
	vRe := []float64{1.0}
	vIm := []float64{2.0}
	twRe := []float64{0.5}
	twIm := []float64{-0.5}

	wantURe := []float64{3.0}
	wantUIm := []float64{4.0}
	wantVRe := []float64{1.0}
	wantVIm := []float64{2.0}
	butterflyGo(wantURe, wantUIm, wantVRe, wantVIm, twRe, twIm)

	ButterflyBlock(uRe, uIm, vRe, vIm, twRe, twIm)

	if math.Abs(uRe[0]-wantURe[0]) > 1e-10 || math.Abs(uIm[0]-wantUIm[0]) > 1e-10 {
		t.Errorf("u = (%f,%f), want (%f,%f)", uRe[0], uIm[0], wantURe[0], wantUIm[0])
	}
	if math.Abs(vRe[0]-wantVRe[0]) > 1e-10 || math.Abs(vIm[0]-wantVIm[0]) > 1e-10 {
		t.Errorf("v = (%f,%f), want (%f,%f)", vRe[0], vIm[0], wantVRe[0], wantVIm[0])
	}
}

func TestButterflyBlock_empty(t *testing.T) {
	ButterflyBlock(nil, nil, nil, nil, nil, nil)
	// Should not panic
}

func TestButterflyBlock_256(t *testing.T) {
	// Large block matching real FFT halfSize
	n := 256
	uRe := make([]float64, n)
	uIm := make([]float64, n)
	vRe := make([]float64, n)
	vIm := make([]float64, n)
	twRe := make([]float64, n)
	twIm := make([]float64, n)

	for i := 0; i < n; i++ {
		theta := -2 * math.Pi * float64(i) / float64(2*n)
		twRe[i] = math.Cos(theta)
		twIm[i] = math.Sin(theta)
		uRe[i] = math.Sin(float64(i) * 0.1)
		uIm[i] = math.Cos(float64(i) * 0.1)
		vRe[i] = math.Sin(float64(i)*0.2 + 1)
		vIm[i] = math.Cos(float64(i)*0.2 + 1)
	}

	wantURe := make([]float64, n)
	wantUIm := make([]float64, n)
	wantVRe := make([]float64, n)
	wantVIm := make([]float64, n)
	copy(wantURe, uRe)
	copy(wantUIm, uIm)
	copy(wantVRe, vRe)
	copy(wantVIm, vIm)
	butterflyGo(wantURe, wantUIm, wantVRe, wantVIm, twRe, twIm)

	ButterflyBlock(uRe, uIm, vRe, vIm, twRe, twIm)

	for i := 0; i < n; i++ {
		if math.Abs(uRe[i]-wantURe[i]) > 1e-10 {
			t.Errorf("uRe[%d] = %g, want %g", i, uRe[i], wantURe[i])
		}
		if math.Abs(uIm[i]-wantUIm[i]) > 1e-10 {
			t.Errorf("uIm[%d] = %g, want %g", i, uIm[i], wantUIm[i])
		}
		if math.Abs(vRe[i]-wantVRe[i]) > 1e-10 {
			t.Errorf("vRe[%d] = %g, want %g", i, vRe[i], wantVRe[i])
		}
		if math.Abs(vIm[i]-wantVIm[i]) > 1e-10 {
			t.Errorf("vIm[%d] = %g, want %g", i, vIm[i], wantVIm[i])
		}
	}
}

func BenchmarkButterflyBlock_256(b *testing.B) {
	n := 256
	uRe := make([]float64, n)
	uIm := make([]float64, n)
	vRe := make([]float64, n)
	vIm := make([]float64, n)
	twRe := make([]float64, n)
	twIm := make([]float64, n)
	for i := 0; i < n; i++ {
		theta := -2 * math.Pi * float64(i) / float64(2*n)
		twRe[i] = math.Cos(theta)
		twIm[i] = math.Sin(theta)
		uRe[i] = float64(i) * 0.1
		vRe[i] = float64(i) * 0.2
	}
	b.ResetTimer()
	for b.Loop() {
		ButterflyBlock(uRe, uIm, vRe, vIm, twRe, twIm)
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
