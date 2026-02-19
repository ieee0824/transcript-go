package acoustic

import (
	"math"
	"math/rand"
	"testing"
)

func TestLogProbBatchMat_MatchesSingle(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	dim := 39
	K := 4
	T := 300

	gmm := NewGMM(K, dim)

	// Generate random features
	xs := make([][]float64, T)
	flatXs := make([]float64, T*dim)
	for i := 0; i < T; i++ {
		xs[i] = make([]float64, dim)
		for d := 0; d < dim; d++ {
			v := rng.NormFloat64()
			xs[i][d] = v
			flatXs[i*dim+d] = v
		}
	}

	// Single-frame reference
	ref := make([]float64, T)
	for i := 0; i < T; i++ {
		ref[i] = gmm.LogProb(xs[i])
	}

	// Batch BLAS
	dst := make([]float64, T)
	ws := NewBatchWorkspace(T, dim, K)
	gmm.LogProbBatchMat(flatXs, T, dim, dst, ws)

	for i := 0; i < T; i++ {
		diff := math.Abs(dst[i] - ref[i])
		if diff > 1e-8 {
			t.Errorf("frame %d: batch=%f, single=%f, diff=%e", i, dst[i], ref[i], diff)
		}
	}
}

func TestLogProbBatchMat_SingleFrame(t *testing.T) {
	dim := 39
	K := 4

	gmm := NewGMM(K, dim)

	rng := rand.New(rand.NewSource(99))
	x := make([]float64, dim)
	for d := range x {
		x[d] = rng.NormFloat64()
	}

	ref := gmm.LogProb(x)

	dst := make([]float64, 1)
	ws := NewBatchWorkspace(1, dim, K)
	gmm.LogProbBatchMat(x, 1, dim, dst, ws)

	if diff := math.Abs(dst[0] - ref); diff > 1e-10 {
		t.Errorf("single frame: batch=%f, single=%f, diff=%e", dst[0], ref, diff)
	}
}

func TestLogProbBatchMat_MultiMix(t *testing.T) {
	// Test with different mixture counts
	for _, K := range []int{1, 2, 4, 8} {
		dim := 13
		T := 50
		gmm := NewGMM(K, dim)

		rng := rand.New(rand.NewSource(int64(K)))
		flatXs := make([]float64, T*dim)
		xs := make([][]float64, T)
		for i := 0; i < T; i++ {
			xs[i] = make([]float64, dim)
			for d := 0; d < dim; d++ {
				v := rng.NormFloat64()
				flatXs[i*dim+d] = v
				xs[i][d] = v
			}
		}

		ref := make([]float64, T)
		for i := 0; i < T; i++ {
			ref[i] = gmm.LogProb(xs[i])
		}

		dst := make([]float64, T)
		ws := NewBatchWorkspace(T, dim, K)
		gmm.LogProbBatchMat(flatXs, T, dim, dst, ws)

		for i := 0; i < T; i++ {
			if diff := math.Abs(dst[i] - ref[i]); diff > 1e-8 {
				t.Errorf("K=%d frame %d: batch=%f, single=%f, diff=%e", K, i, dst[i], ref[i], diff)
			}
		}
	}
}

func BenchmarkGMM_LogProbBatchMat_300x4(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	dim := 39
	K := 4
	T := 300

	gmm := NewGMM(K, dim)
	flatXs := make([]float64, T*dim)
	for i := range flatXs {
		flatXs[i] = rng.NormFloat64()
	}
	dst := make([]float64, T)
	ws := NewBatchWorkspace(T, dim, K)

	b.ResetTimer()
	for b.Loop() {
		gmm.LogProbBatchMat(flatXs, T, dim, dst, ws)
	}
}

func BenchmarkGMM_LogProbBatch_300x4(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	dim := 39
	K := 4
	T := 300

	gmm := NewGMM(K, dim)
	xs := make([][]float64, T)
	for i := 0; i < T; i++ {
		xs[i] = make([]float64, dim)
		for d := 0; d < dim; d++ {
			xs[i][d] = rng.NormFloat64()
		}
	}
	dst := make([]float64, T)

	b.ResetTimer()
	for b.Loop() {
		gmm.LogProbBatch(xs, dst)
	}
}
