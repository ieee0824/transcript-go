package acoustic

import (
	"bytes"
	"math"
	"math/rand"
	"testing"
)

func TestAllPhonemes(t *testing.T) {
	phonemes := AllPhonemes()
	if len(phonemes) != 29 {
		t.Errorf("len(AllPhonemes) = %d, want 29", len(phonemes))
	}
	// Check no duplicates
	seen := make(map[Phoneme]bool)
	for _, p := range phonemes {
		if seen[p] {
			t.Errorf("duplicate phoneme: %s", p)
		}
		seen[p] = true
	}
}

func TestGaussianLogProb(t *testing.T) {
	g := Gaussian{
		Mean:      []float64{0.0},
		Variance:  []float64{1.0},
		LogWeight: 0.0,
	}
	g.Precompute()

	// Standard normal at x=0: log(1/sqrt(2π)) ≈ -0.9189
	lp := g.LogProb([]float64{0.0})
	expected := -0.5 * math.Log(2*math.Pi)
	if math.Abs(lp-expected) > 1e-6 {
		t.Errorf("LogProb(0) = %f, want %f", lp, expected)
	}

	// At x=0 should be higher than at x=5
	lp5 := g.LogProb([]float64{5.0})
	if lp5 >= lp {
		t.Errorf("LogProb(5) = %f >= LogProb(0) = %f", lp5, lp)
	}
}

func TestGMMLogProb(t *testing.T) {
	gmm := NewGMMWithParams(
		[][]float64{{0.0}, {5.0}},
		[][]float64{{1.0}, {1.0}},
		[]float64{math.Log(0.5), math.Log(0.5)},
	)

	// At x=0, first component dominates
	lp0 := gmm.LogProb([]float64{0.0})
	// At x=5, second component dominates
	lp5 := gmm.LogProb([]float64{5.0})
	// At x=2.5, mixture of both
	lp25 := gmm.LogProb([]float64{2.5})

	// Both log probs should be finite
	if math.IsNaN(lp0) || math.IsInf(lp0, 0) {
		t.Errorf("LogProb(0) = %f (not finite)", lp0)
	}
	if math.IsNaN(lp5) || math.IsInf(lp5, 0) {
		t.Errorf("LogProb(5) = %f (not finite)", lp5)
	}

	// x=0 and x=5 should have similar prob (symmetric mixture)
	if math.Abs(lp0-lp5) > 0.1 {
		t.Errorf("LogProb(0)=%f and LogProb(5)=%f should be similar", lp0, lp5)
	}

	// x=2.5 should be lower than x=0 (between the modes)
	if lp25 > lp0 {
		t.Errorf("LogProb(2.5)=%f > LogProb(0)=%f", lp25, lp0)
	}
}

func TestNewPhonemeHMM(t *testing.T) {
	hmm := NewPhonemeHMM(PhonA, 13, 2)
	if hmm.Phoneme != PhonA {
		t.Errorf("Phoneme = %s, want a", hmm.Phoneme)
	}
	// Entry state should be nil
	if hmm.States[0] != nil {
		t.Error("entry state should be nil")
	}
	// Exit state should be nil
	if hmm.States[4] != nil {
		t.Error("exit state should be nil")
	}
	// Emitting states should have GMMs
	for i := 1; i <= 3; i++ {
		if hmm.States[i] == nil {
			t.Errorf("state %d should not be nil", i)
		}
		if hmm.States[i].GMM.Dim != 13 {
			t.Errorf("state %d dim = %d, want 13", i, hmm.States[i].GMM.Dim)
		}
		if len(hmm.States[i].GMM.Components) != 2 {
			t.Errorf("state %d components = %d, want 2", i, len(hmm.States[i].GMM.Components))
		}
	}
}

func TestForwardBackward(t *testing.T) {
	// Create a simple 1D HMM
	dim := 1
	hmm := NewPhonemeHMM(PhonA, dim, 1)

	// Set deterministic GMMs: state 1→mean 0, state 2→mean 3, state 3→mean 6
	for i := 1; i <= 3; i++ {
		hmm.States[i].GMM = NewGMMWithParams(
			[][]float64{{float64((i - 1) * 3)}},
			[][]float64{{1.0}},
			[]float64{0.0},
		)
	}

	obs := [][]float64{{0.0}, {0.5}, {3.0}, {3.5}, {6.0}, {6.5}}

	alpha := Forward(hmm, obs)
	beta := Backward(hmm, obs)

	// Total log-likelihood from forward and backward should agree
	llForward := totalLogLikelihood(alpha)

	llBackward := math.Inf(-1)
	for j := 1; j <= NumEmittingStates; j++ {
		val := alpha[0][j] + beta[0][j]
		if llBackward == math.Inf(-1) {
			llBackward = val
		} else {
			llBackward = math.Log(math.Exp(llBackward) + math.Exp(val))
		}
	}

	if math.Abs(llForward-llBackward) > 0.5 {
		t.Errorf("forward LL = %f, backward LL = %f, differ too much", llForward, llBackward)
	}

	if math.IsNaN(llForward) || math.IsInf(llForward, -1) {
		t.Errorf("forward LL is invalid: %f", llForward)
	}
}

func TestBaumWelch_LikelihoodImproves(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	dim := 2

	// Generate synthetic training data:
	// Sequences following a pattern: state1→mean[0,0], state2→mean[3,3], state3→mean[6,6]
	numSeqs := 50
	seqLen := 6
	sequences := make([][][]float64, numSeqs)
	for s := 0; s < numSeqs; s++ {
		seq := make([][]float64, seqLen)
		for t := 0; t < seqLen; t++ {
			// Roughly: first 2 frames -> state 1, next 2 -> state 2, last 2 -> state 3
			stateCenter := float64(t/2) * 3.0
			seq[t] = []float64{stateCenter + rng.NormFloat64()*0.5, stateCenter + rng.NormFloat64()*0.5}
		}
		sequences[s] = seq
	}

	hmm := NewPhonemeHMM(PhonA, dim, 1)

	// Compute initial log-likelihood
	initialLL := 0.0
	for _, obs := range sequences {
		alpha := Forward(hmm, obs)
		initialLL += totalLogLikelihood(alpha)
	}

	cfg := TrainingConfig{
		MaxIterations:     10,
		ConvergenceThresh: 0.001,
		MinVariance:       0.01,
	}
	err := TrainPhoneme(hmm, sequences, cfg)
	if err != nil {
		t.Fatalf("TrainPhoneme error: %v", err)
	}

	// Compute final log-likelihood
	finalLL := 0.0
	for _, obs := range sequences {
		alpha := Forward(hmm, obs)
		finalLL += totalLogLikelihood(alpha)
	}

	if finalLL <= initialLL {
		t.Errorf("training did not improve LL: initial=%f, final=%f", initialLL, finalLL)
	}
}

func TestModelSaveLoad(t *testing.T) {
	am := NewAcousticModel(13, 2)

	var buf bytes.Buffer
	if err := am.Save(&buf); err != nil {
		t.Fatalf("Save error: %v", err)
	}

	loaded, err := Load(&buf)
	if err != nil {
		t.Fatalf("Load error: %v", err)
	}

	if loaded.FeatureDim != am.FeatureDim {
		t.Errorf("FeatureDim = %d, want %d", loaded.FeatureDim, am.FeatureDim)
	}
	if loaded.NumMix != am.NumMix {
		t.Errorf("NumMix = %d, want %d", loaded.NumMix, am.NumMix)
	}
	if len(loaded.Phonemes) != len(am.Phonemes) {
		t.Errorf("len(Phonemes) = %d, want %d", len(loaded.Phonemes), len(am.Phonemes))
	}

	// Verify a specific phoneme's GMM parameters survived round-trip
	for p, origHMM := range am.Phonemes {
		loadedHMM, ok := loaded.Phonemes[p]
		if !ok {
			t.Errorf("missing phoneme %s after load", p)
			continue
		}
		for s := 1; s <= NumEmittingStates; s++ {
			origGMM := origHMM.States[s].GMM
			loadedGMM := loadedHMM.States[s].GMM
			if origGMM.Dim != loadedGMM.Dim {
				t.Errorf("phoneme %s state %d: dim %d != %d", p, s, origGMM.Dim, loadedGMM.Dim)
			}
			for m := range origGMM.Components {
				for d := range origGMM.Components[m].Mean {
					if math.Abs(origGMM.Components[m].Mean[d]-loadedGMM.Components[m].Mean[d]) > 1e-10 {
						t.Errorf("phoneme %s state %d mix %d mean[%d]: %f != %f",
							p, s, m, d, origGMM.Components[m].Mean[d], loadedGMM.Components[m].Mean[d])
					}
				}
			}
		}
	}
}
