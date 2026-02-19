package acoustic

import "math"

// FrameLikelihood scores features against all monophone emitting states.
// For each frame, it takes the maximum GMM log-prob across all monophone
// emitting states, then sums across frames.
// This is used as the scoring function for VTLN grid search.
func FrameLikelihood(am *AcousticModel, features [][]float64) float64 {
	if len(features) == 0 {
		return math.Inf(-1)
	}

	// Collect all emitting-state GMMs from monophones.
	var gmms []*GMM
	for _, hmm := range am.Phonemes {
		for s := 1; s <= NumEmittingStates; s++ {
			if hmm.States[s] != nil && hmm.States[s].GMM != nil {
				gmms = append(gmms, hmm.States[s].GMM)
			}
		}
	}
	if len(gmms) == 0 {
		return math.Inf(-1)
	}

	total := 0.0
	for _, obs := range features {
		best := math.Inf(-1)
		for _, g := range gmms {
			lp := g.LogProb(obs)
			if lp > best {
				best = lp
			}
		}
		total += best
	}
	return total
}
