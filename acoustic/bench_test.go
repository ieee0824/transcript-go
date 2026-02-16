package acoustic

import (
	"math/rand"
	"testing"
)

func randomObs(dim int) []float64 {
	obs := make([]float64, dim)
	for i := range obs {
		obs[i] = rand.NormFloat64()
	}
	return obs
}

func randomObsSeq(T, dim int) [][]float64 {
	seq := make([][]float64, T)
	for t := range seq {
		seq[t] = randomObs(dim)
	}
	return seq
}

func BenchmarkGMM_LogProb_1mix_39dim(b *testing.B) {
	gmm := NewGMM(1, 39)
	obs := randomObs(39)
	b.ResetTimer()
	for b.Loop() {
		gmm.LogProb(obs)
	}
}

func BenchmarkGMM_LogProb_4mix_39dim(b *testing.B) {
	gmm := NewGMM(4, 39)
	obs := randomObs(39)
	b.ResetTimer()
	for b.Loop() {
		gmm.LogProb(obs)
	}
}

func BenchmarkGMM_LogProb_16mix_39dim(b *testing.B) {
	gmm := NewGMM(16, 39)
	obs := randomObs(39)
	b.ResetTimer()
	for b.Loop() {
		gmm.LogProb(obs)
	}
}

func BenchmarkForward_100frames(b *testing.B) {
	hmm := NewPhonemeHMM(PhonA, 39, 4)
	obs := randomObsSeq(100, 39)
	b.ResetTimer()
	for b.Loop() {
		Forward(hmm, obs)
	}
}

func BenchmarkForward_500frames(b *testing.B) {
	hmm := NewPhonemeHMM(PhonA, 39, 4)
	obs := randomObsSeq(500, 39)
	b.ResetTimer()
	for b.Loop() {
		Forward(hmm, obs)
	}
}

func BenchmarkBaumWelch_10seq_50frames(b *testing.B) {
	seqs := make([][][]float64, 10)
	for i := range seqs {
		seqs[i] = randomObsSeq(50, 39)
	}
	cfg := TrainingConfig{
		MaxIterations:     3,
		ConvergenceThresh: 0.001,
		MinVariance:       0.01,
	}
	b.ResetTimer()
	for b.Loop() {
		h := NewPhonemeHMM(PhonA, 39, 2)
		TrainPhoneme(h, seqs, cfg)
	}
}
