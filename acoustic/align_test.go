package acoustic

import (
	"testing"
)

// makeTestModel creates a simple acoustic model with phonemes whose GMMs
// have distinct means so that forced alignment can find clear boundaries.
func makeTestModel(dim int, phonemes []Phoneme, means []float64) *AcousticModel {
	am := &AcousticModel{
		Phonemes:   make(map[Phoneme]*PhonemeHMM),
		FeatureDim: dim,
		NumMix:     1,
	}
	for i, ph := range phonemes {
		hmm := NewPhonemeHMM(ph, dim, 1)
		// Set GMM mean for all emitting states
		for s := 1; s <= NumEmittingStates; s++ {
			for d := 0; d < dim; d++ {
				hmm.States[s].GMM.Components[0].Mean[d] = means[i]
				hmm.States[s].GMM.Components[0].Variance[d] = 0.5
			}
			hmm.States[s].GMM.Components[0].Precompute()
			hmm.States[s].GMM.PrecomputeSoA()
		}
		am.Phonemes[ph] = hmm
	}
	return am
}

// makeFeatures creates T frames of dim-dimensional features all set to val.
func makeFeatures(T, dim int, val float64) [][]float64 {
	f := make([][]float64, T)
	for t := range f {
		f[t] = make([]float64, dim)
		for d := range f[t] {
			f[t][d] = val
		}
	}
	return f
}

func TestForcedAlign_ThreePhonemes(t *testing.T) {
	dim := 4
	phonemes := []Phoneme{PhonA, PhonK, PhonI}
	means := []float64{0.0, 5.0, 10.0}
	am := makeTestModel(dim, phonemes, means)

	// Create features: 10 frames near 0.0, 10 near 5.0, 10 near 10.0
	features := make([][]float64, 30)
	for t := 0; t < 10; t++ {
		features[t] = make([]float64, dim)
		for d := range features[t] {
			features[t][d] = 0.1
		}
	}
	for t := 10; t < 20; t++ {
		features[t] = make([]float64, dim)
		for d := range features[t] {
			features[t][d] = 5.1
		}
	}
	for t := 20; t < 30; t++ {
		features[t] = make([]float64, dim)
		for d := range features[t] {
			features[t][d] = 10.1
		}
	}

	alignments, err := ForcedAlign(am, phonemes, features)
	if err != nil {
		t.Fatalf("ForcedAlign error: %v", err)
	}

	if len(alignments) != 3 {
		t.Fatalf("expected 3 alignments, got %d", len(alignments))
	}

	// Verify phoneme order
	for i, a := range alignments {
		if a.Phoneme != phonemes[i] {
			t.Errorf("alignment[%d]: expected phoneme %s, got %s", i, phonemes[i], a.Phoneme)
		}
	}

	// Verify boundaries are contiguous
	if alignments[0].StartFrame != 0 {
		t.Errorf("first alignment should start at 0, got %d", alignments[0].StartFrame)
	}
	if alignments[2].EndFrame != 30 {
		t.Errorf("last alignment should end at 30, got %d", alignments[2].EndFrame)
	}
	for i := 1; i < len(alignments); i++ {
		if alignments[i].StartFrame != alignments[i-1].EndFrame {
			t.Errorf("gap between alignment %d and %d: %d != %d",
				i-1, i, alignments[i-1].EndFrame, alignments[i].StartFrame)
		}
	}

	// Verify boundaries are roughly correct (each phoneme gets ~10 frames)
	for i, a := range alignments {
		dur := a.EndFrame - a.StartFrame
		if dur < 3 {
			t.Errorf("alignment[%d] (%s): duration %d too short", i, a.Phoneme, dur)
		}
		t.Logf("alignment[%d] %s: frames [%d, %d) dur=%d", i, a.Phoneme, a.StartFrame, a.EndFrame, dur)
	}
}

func TestForcedAlign_SinglePhoneme(t *testing.T) {
	dim := 4
	am := makeTestModel(dim, []Phoneme{PhonA}, []float64{0.0})
	features := makeFeatures(20, dim, 0.1)

	alignments, err := ForcedAlign(am, []Phoneme{PhonA}, features)
	if err != nil {
		t.Fatalf("ForcedAlign error: %v", err)
	}

	if len(alignments) != 1 {
		t.Fatalf("expected 1 alignment, got %d", len(alignments))
	}
	if alignments[0].StartFrame != 0 || alignments[0].EndFrame != 20 {
		t.Errorf("expected [0, 20), got [%d, %d)", alignments[0].StartFrame, alignments[0].EndFrame)
	}
}

func TestForcedAlign_TooFewFrames(t *testing.T) {
	dim := 4
	am := makeTestModel(dim, []Phoneme{PhonA, PhonK, PhonI}, []float64{0, 5, 10})
	features := makeFeatures(2, dim, 0.0) // 2 frames < 3 phonemes

	_, err := ForcedAlign(am, []Phoneme{PhonA, PhonK, PhonI}, features)
	if err == nil {
		t.Error("expected error for too few frames, got nil")
	}
}

func TestForcedAlign_MissingPhoneme(t *testing.T) {
	dim := 4
	am := makeTestModel(dim, []Phoneme{PhonA}, []float64{0.0})
	features := makeFeatures(10, dim, 0.0)

	_, err := ForcedAlign(am, []Phoneme{PhonA, PhonZ}, features) // PhonZ not in model
	if err == nil {
		t.Error("expected error for missing phoneme, got nil")
	}
}
