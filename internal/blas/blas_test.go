package blas

import (
	"math"
	"math/rand"
	"testing"
)

func TestDgemm_Identity(t *testing.T) {
	// A(2x3) * I(3x3) = A(2x3)
	a := []float64{1, 2, 3, 4, 5, 6}
	b := []float64{1, 0, 0, 0, 1, 0, 0, 0, 1}
	c := make([]float64, 6)

	Dgemm(false, false, 2, 3, 3, 1.0, a, 3, b, 3, 0.0, c, 3)

	for i, want := range a {
		if math.Abs(c[i]-want) > 1e-12 {
			t.Errorf("c[%d] = %f, want %f", i, c[i], want)
		}
	}
}

func TestDgemm_Small(t *testing.T) {
	// A(2x3) * B(3x2) = C(2x2)
	a := []float64{1, 2, 3, 4, 5, 6}
	b := []float64{7, 8, 9, 10, 11, 12}
	c := make([]float64, 4)

	Dgemm(false, false, 2, 2, 3, 1.0, a, 3, b, 2, 0.0, c, 2)

	// C[0,0] = 1*7 + 2*9 + 3*11 = 7+18+33 = 58
	// C[0,1] = 1*8 + 2*10 + 3*12 = 8+20+36 = 64
	// C[1,0] = 4*7 + 5*9 + 6*11 = 28+45+66 = 139
	// C[1,1] = 4*8 + 5*10 + 6*12 = 32+50+72 = 154
	want := []float64{58, 64, 139, 154}
	for i := range want {
		if math.Abs(c[i]-want[i]) > 1e-10 {
			t.Errorf("c[%d] = %f, want %f", i, c[i], want[i])
		}
	}
}

func TestDgemm_TransB(t *testing.T) {
	// A(2x3) * B^T where B is (2x3) stored row-major → B^T is (3x2)
	// Result: C(2x2)
	a := []float64{1, 2, 3, 4, 5, 6}
	b := []float64{7, 9, 11, 8, 10, 12} // B(2x3), B^T(3x2) = [[7,8],[9,10],[11,12]]
	c := make([]float64, 4)

	Dgemm(false, true, 2, 2, 3, 1.0, a, 3, b, 3, 0.0, c, 2)

	// Same result as TestDgemm_Small since B^T matches
	want := []float64{58, 64, 139, 154}
	for i := range want {
		if math.Abs(c[i]-want[i]) > 1e-10 {
			t.Errorf("c[%d] = %f, want %f", i, c[i], want[i])
		}
	}
}

func TestDgemm_AlphaBeta(t *testing.T) {
	// C = 2*A*B + 3*C
	a := []float64{1, 2, 3, 4}
	b := []float64{5, 6, 7, 8}
	c := []float64{1, 1, 1, 1}

	Dgemm(false, false, 2, 2, 2, 2.0, a, 2, b, 2, 3.0, c, 2)

	// A*B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
	// 2*A*B + 3*C = [[38+3, 44+3], [86+3, 100+3]] = [[41, 47], [89, 103]]
	want := []float64{41, 47, 89, 103}
	for i := range want {
		if math.Abs(c[i]-want[i]) > 1e-10 {
			t.Errorf("c[%d] = %f, want %f", i, c[i], want[i])
		}
	}
}

func TestDgemm_GMMSized(t *testing.T) {
	// Typical GMM batch: (300x39) * (39x4)^T where B is (4x39)
	rng := rand.New(rand.NewSource(42))
	T, D, K := 300, 39, 4

	a := make([]float64, T*D) // X^2
	b := make([]float64, K*D) // invVar (K x D)
	for i := range a {
		a[i] = rng.Float64()
	}
	for i := range b {
		b[i] = rng.Float64()
	}

	// Compute with Dgemm: C = A * B^T
	c := make([]float64, T*K)
	Dgemm(false, true, T, K, D, 1.0, a, D, b, D, 0.0, c, K)

	// Verify with naive computation
	for i := 0; i < T; i++ {
		for j := 0; j < K; j++ {
			sum := 0.0
			for p := 0; p < D; p++ {
				sum += a[i*D+p] * b[j*D+p]
			}
			if math.Abs(c[i*K+j]-sum) > 1e-8 {
				t.Errorf("c[%d,%d] = %f, want %f (diff=%e)", i, j, c[i*K+j], sum, c[i*K+j]-sum)
			}
		}
	}
}

func BenchmarkDgemm_300x39x4(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	T, D, K := 300, 39, 4
	a := make([]float64, T*D)
	bm := make([]float64, K*D)
	for i := range a {
		a[i] = rng.Float64()
	}
	for i := range bm {
		bm[i] = rng.Float64()
	}
	c := make([]float64, T*K)

	b.ResetTimer()
	for b.Loop() {
		Dgemm(false, true, T, K, D, 1.0, a, D, bm, D, 0.0, c, K)
	}
}
