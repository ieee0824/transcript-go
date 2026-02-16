package acoustic

import (
	"math"
	"github.com/ieee0824/transcript-go/internal/mathutil"
)

// PhonemeHMM represents a left-to-right HMM for a single phoneme.
// States: [0]=entry (non-emitting), [1..3]=emitting, [4]=exit (non-emitting).
// Transitions are left-to-right only with self-loops on emitting states.
type PhonemeHMM struct {
	Phoneme  Phoneme
	States   []*GMMState
	TransLog [][]float64 // [NumStatesPerPhoneme][NumStatesPerPhoneme] log transition probs
}

// GMMState wraps a GMM for an emitting state.
type GMMState struct {
	GMM *GMM
}

// NewPhonemeHMM creates a new left-to-right HMM for the given phoneme.
// featureDim: dimension of observation vectors
// numMix: number of GMM components per emitting state
func NewPhonemeHMM(phoneme Phoneme, featureDim, numMix int) *PhonemeHMM {
	hmm := &PhonemeHMM{
		Phoneme:  phoneme,
		States:   make([]*GMMState, NumStatesPerPhoneme),
		TransLog: mathutil.NewMatFill(NumStatesPerPhoneme, NumStatesPerPhoneme, mathutil.LogZero),
	}

	// Create GMMs for emitting states (indices 1, 2, 3)
	for i := 1; i <= NumEmittingStates; i++ {
		hmm.States[i] = &GMMState{GMM: NewGMM(numMix, featureDim)}
	}
	// Non-emitting states (entry=0, exit=4) have nil GMM
	hmm.States[0] = nil
	hmm.States[NumStatesPerPhoneme-1] = nil

	// Initialize transition probabilities (left-to-right)
	// Entry -> first emitting state
	hmm.TransLog[0][1] = 0.0 // log(1.0)

	// Emitting states: self-loop (0.5) and next state (0.5)
	logHalf := math.Log(0.5)
	for i := 1; i <= NumEmittingStates; i++ {
		hmm.TransLog[i][i] = logHalf // self-loop
		if i < NumStatesPerPhoneme-1 {
			hmm.TransLog[i][i+1] = logHalf // forward
		}
	}

	return hmm
}

// LogLikelihood computes log P(observation | state) for an emitting state.
func (h *PhonemeHMM) LogLikelihood(stateIdx int, obs []float64) float64 {
	if stateIdx <= 0 || stateIdx > NumEmittingStates {
		return mathutil.LogZero
	}
	if h.States[stateIdx] == nil {
		return mathutil.LogZero
	}
	return h.States[stateIdx].GMM.LogProb(obs)
}

// IsEmitting returns true if the state index corresponds to an emitting state.
func IsEmitting(stateIdx int) bool {
	return stateIdx >= 1 && stateIdx <= NumEmittingStates
}
