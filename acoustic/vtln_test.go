package acoustic

import (
	"math"
	"testing"
)

func TestFrameLikelihood(t *testing.T) {
	dim := 39
	numMix := 2
	am := NewAcousticModel(dim, numMix)

	// Generate simple features: 10 frames of zeros
	features := make([][]float64, 10)
	for i := range features {
		features[i] = make([]float64, dim)
	}

	score := FrameLikelihood(am, features)
	if math.IsNaN(score) || math.IsInf(score, 0) {
		t.Errorf("FrameLikelihood returned non-finite: %f", score)
	}
	// Score should be negative (log probabilities)
	if score > 0 {
		t.Errorf("FrameLikelihood = %f, expected negative", score)
	}
}

func TestFrameLikelihood_Empty(t *testing.T) {
	am := NewAcousticModel(39, 2)
	score := FrameLikelihood(am, nil)
	if !math.IsInf(score, -1) {
		t.Errorf("FrameLikelihood(nil) = %f, want -Inf", score)
	}
}
