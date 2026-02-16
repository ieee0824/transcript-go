package acoustic

import (
	"fmt"
	"math"

	"github.com/ieee0824/transcript-go/internal/mathutil"
)

// PhonemeAlignment holds forced alignment result for one phoneme.
type PhonemeAlignment struct {
	Phoneme    Phoneme
	StartFrame int // inclusive
	EndFrame   int // exclusive
}

// ForcedAlign performs Viterbi forced alignment of a feature sequence
// against a known phoneme sequence using the given acoustic model.
// Returns per-phoneme frame boundaries.
func ForcedAlign(am *AcousticModel, phonemes []Phoneme, features [][]float64) ([]PhonemeAlignment, error) {
	T := len(features)
	N := len(phonemes)
	if N == 0 {
		return nil, fmt.Errorf("empty phoneme sequence")
	}
	if T < N {
		return nil, fmt.Errorf("too few frames (%d) for %d phonemes", T, N)
	}

	// Validate all phonemes exist in model and collect HMMs
	hmms := make([]*PhonemeHMM, N)
	for i, ph := range phonemes {
		h, ok := am.Phonemes[ph]
		if !ok {
			return nil, fmt.Errorf("phoneme %q not in acoustic model", ph)
		}
		if h.States[1] == nil {
			return nil, fmt.Errorf("phoneme %q has no trained emitting states", ph)
		}
		hmms[i] = h
	}

	S := N * NumEmittingStates // total composite states

	// Precompute emission log-likelihoods: emit[t][j]
	// Iterate state-outer, frame-inner for cache locality (GMM SoA pattern)
	emit := mathutil.NewMat(T, S)
	for p := 0; p < N; p++ {
		hmm := hmms[p]
		for s := 1; s <= NumEmittingStates; s++ {
			j := p*NumEmittingStates + (s - 1)
			gmm := hmm.States[s].GMM
			for t := 0; t < T; t++ {
				emit[t][j] = gmm.LogProb(features[t])
			}
		}
	}

	// Viterbi with double-buffered score vectors
	prev := mathutil.NewVecFill(S, mathutil.LogZero)
	curr := mathutil.NewVecFill(S, mathutil.LogZero)

	// Backpointer matrix: bp[t][j] = predecessor flat index
	bp := make([][]int32, T)
	for t := range bp {
		bp[t] = make([]int32, S)
	}

	// Precompute exit transition scores for each phoneme.
	// Baum-Welch on isolated segments doesn't estimate exit transitions properly
	// (TransLog[3][4] often becomes LogZero), so we use a floor.
	exitTrans := make([]float64, N)
	defaultExit := math.Log(0.5)
	for p := 0; p < N; p++ {
		et := hmms[p].TransLog[NumEmittingStates][NumStatesPerPhoneme-1]
		if et <= mathutil.LogZero+1 {
			et = defaultExit
		}
		exitTrans[p] = et
	}

	// Initialize t=0: only (p=0, s=1) reachable
	prev[0] = hmms[0].TransLog[0][1] + emit[0][0]

	// Recursion
	for t := 1; t < T; t++ {
		mathutil.FillVec(curr, mathutil.LogZero)

		for j := 0; j < S; j++ {
			p := j / NumEmittingStates
			s := j%NumEmittingStates + 1 // emitting state 1, 2, or 3
			hmm := hmms[p]

			bestScore := mathutil.LogZero
			bestPrev := int32(0)

			// Candidate 1: self-loop (p, s) -> (p, s)
			score := prev[j] + hmm.TransLog[s][s]
			if score > bestScore {
				bestScore = score
				bestPrev = int32(j)
			}

			// Candidate 2: intra-phoneme forward (p, s-1) -> (p, s)
			if s >= 2 {
				prevJ := p*NumEmittingStates + (s - 2)
				score = prev[prevJ] + hmm.TransLog[s-1][s]
				if score > bestScore {
					bestScore = score
					bestPrev = int32(prevJ)
				}
			}

			// Candidate 3: cross-phoneme (p-1, 3) -> (p, 1)
			if s == 1 && p >= 1 {
				prevJ := (p-1)*NumEmittingStates + (NumEmittingStates - 1) // last emitting state of prev phoneme
				score = prev[prevJ] + exitTrans[p-1] // entry trans [0][1] = 0, omitted
				if score > bestScore {
					bestScore = score
					bestPrev = int32(prevJ)
				}
			}

			if bestScore > mathutil.LogZero+1 {
				curr[j] = bestScore + emit[t][j]
			}
			bp[t][j] = bestPrev
		}

		prev, curr = curr, prev
	}

	// Termination: best among last phoneme's emitting states
	bestJ := -1
	bestScore := mathutil.LogZero
	for s := 0; s < NumEmittingStates; s++ {
		j := (N-1)*NumEmittingStates + s
		if prev[j] > bestScore {
			bestScore = prev[j]
			bestJ = j
		}
	}
	if bestJ < 0 || bestScore <= mathutil.LogZero+1 {
		return nil, fmt.Errorf("forced alignment failed: no valid path")
	}

	// Backtrace
	path := make([]int, T)
	path[T-1] = bestJ
	for t := T - 1; t > 0; t-- {
		path[t-1] = int(bp[t][path[t]])
	}

	// Extract phoneme boundaries from path
	result := make([]PhonemeAlignment, 0, N)
	currentP := path[0] / NumEmittingStates
	startFrame := 0
	for t := 1; t < T; t++ {
		p := path[t] / NumEmittingStates
		if p != currentP {
			result = append(result, PhonemeAlignment{
				Phoneme:    phonemes[currentP],
				StartFrame: startFrame,
				EndFrame:   t,
			})
			currentP = p
			startFrame = t
		}
	}
	// Final phoneme
	result = append(result, PhonemeAlignment{
		Phoneme:    phonemes[currentP],
		StartFrame: startFrame,
		EndFrame:   T,
	})

	return result, nil
}
