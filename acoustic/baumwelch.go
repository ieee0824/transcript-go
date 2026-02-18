package acoustic

import (
	"fmt"
	"math"
	"github.com/ieee0824/transcript-go/internal/mathutil"
)

// TrainingConfig holds Baum-Welch training parameters.
type TrainingConfig struct {
	MaxIterations     int
	ConvergenceThresh float64 // log-likelihood improvement threshold
	MinVariance       float64 // variance floor
}

// DefaultTrainingConfig returns reasonable default training parameters.
func DefaultTrainingConfig() TrainingConfig {
	return TrainingConfig{
		MaxIterations:     20,
		ConvergenceThresh: 0.01,
		MinVariance:       0.01,
	}
}

// Forward computes the forward variable alpha[t][j] in log domain.
// alpha[t][j] = log P(o_1..o_t, q_t=j | model)
// Only emitting states (indices 1..NumEmittingStates) have valid values.
func Forward(hmm *PhonemeHMM, obs [][]float64) [][]float64 {
	T := len(obs)
	N := NumStatesPerPhoneme
	alpha := mathutil.NewMatFill(T, N, mathutil.LogZero)

	// t=0: entry state (0) transitions to emitting states
	for j := 1; j <= NumEmittingStates; j++ {
		if hmm.TransLog[0][j] > mathutil.LogZero+1 {
			alpha[0][j] = hmm.TransLog[0][j] + hmm.LogLikelihood(j, obs[0])
		}
	}

	// Recurse: t=1..T-1
	for t := 1; t < T; t++ {
		for j := 1; j <= NumEmittingStates; j++ {
			logSum := mathutil.LogZero
			// Transitions from emitting states
			for i := 1; i <= NumEmittingStates; i++ {
				if alpha[t-1][i] > mathutil.LogZero+1 && hmm.TransLog[i][j] > mathutil.LogZero+1 {
					logSum = mathutil.LogAdd(logSum, alpha[t-1][i]+hmm.TransLog[i][j])
				}
			}
			if logSum > mathutil.LogZero+1 {
				alpha[t][j] = logSum + hmm.LogLikelihood(j, obs[t])
			}
		}
	}
	return alpha
}

// Backward computes the backward variable beta[t][j] in log domain.
// beta[t][j] = log P(o_{t+1}..o_T, exit | q_t=j, model)
// At T-1, beta includes the exit transition: beta[T-1][i] = a(i, exit).
func Backward(hmm *PhonemeHMM, obs [][]float64) [][]float64 {
	T := len(obs)
	N := NumStatesPerPhoneme
	beta := mathutil.NewMatFill(T, N, mathutil.LogZero)

	// t=T-1: model must exit after the last observation.
	// beta[T-1][i] = a(i, exit) so that P(O|λ) includes exit probability.
	exitJ := NumStatesPerPhoneme - 1
	for i := 1; i <= NumEmittingStates; i++ {
		beta[T-1][i] = hmm.TransLog[i][exitJ]
	}

	// Recurse: t=T-2..0
	for t := T - 2; t >= 0; t-- {
		for i := 1; i <= NumEmittingStates; i++ {
			logSum := mathutil.LogZero
			for j := 1; j <= NumEmittingStates; j++ {
				if hmm.TransLog[i][j] > mathutil.LogZero+1 && beta[t+1][j] > mathutil.LogZero+1 {
					logSum = mathutil.LogAdd(logSum,
						hmm.TransLog[i][j]+hmm.LogLikelihood(j, obs[t+1])+beta[t+1][j])
				}
			}
			beta[t][i] = logSum
		}
	}
	return beta
}

// totalLogLikelihood computes the total log-likelihood from forward variables.
func totalLogLikelihood(alpha [][]float64) float64 {
	T := len(alpha)
	ll := mathutil.LogZero
	for j := 1; j <= NumEmittingStates; j++ {
		ll = mathutil.LogAdd(ll, alpha[T-1][j])
	}
	return ll
}

// computeEmissions precomputes emission log-likelihoods into the emit buffer.
// emit[t][s] = log P(obs[t] | state s+1) for s = 0..NumEmittingStates-1.
// Iterates state-outer, frame-inner to keep GMM SoA data in cache.
func computeEmissions(hmm *PhonemeHMM, obs [][]float64, emit [][]float64) {
	for s := 0; s < NumEmittingStates; s++ {
		gmm := hmm.States[s+1].GMM
		for t, o := range obs {
			emit[t][s] = gmm.LogProb(o)
		}
	}
}

// forwardWithEmit computes forward variables using pre-computed emissions into alpha.
// alpha must be pre-allocated with at least T rows of NumStatesPerPhoneme columns.
func forwardWithEmit(hmm *PhonemeHMM, emit [][]float64, T int, alpha [][]float64) {
	N := NumStatesPerPhoneme
	for t := 0; t < T; t++ {
		for j := 0; j < N; j++ {
			alpha[t][j] = mathutil.LogZero
		}
	}

	for j := 1; j <= NumEmittingStates; j++ {
		if hmm.TransLog[0][j] > mathutil.LogZero+1 {
			alpha[0][j] = hmm.TransLog[0][j] + emit[0][j-1]
		}
	}

	for t := 1; t < T; t++ {
		for j := 1; j <= NumEmittingStates; j++ {
			logSum := mathutil.LogZero
			for i := 1; i <= NumEmittingStates; i++ {
				logSum = mathutil.LogAdd(logSum, alpha[t-1][i]+hmm.TransLog[i][j])
			}
			if logSum > mathutil.LogZero+1 {
				alpha[t][j] = logSum + emit[t][j-1]
			}
		}
	}
}

// backwardWithEmit computes backward variables using pre-computed emissions into beta.
// beta must be pre-allocated with at least T rows of NumStatesPerPhoneme columns.
func backwardWithEmit(hmm *PhonemeHMM, emit [][]float64, T int, beta [][]float64) {
	N := NumStatesPerPhoneme
	for t := 0; t < T; t++ {
		for j := 0; j < N; j++ {
			beta[t][j] = mathutil.LogZero
		}
	}

	// β(T-1, i) = a(i, exit): model must exit after last observation.
	exitJ := NumStatesPerPhoneme - 1
	for i := 1; i <= NumEmittingStates; i++ {
		beta[T-1][i] = hmm.TransLog[i][exitJ]
	}

	for t := T - 2; t >= 0; t-- {
		for i := 1; i <= NumEmittingStates; i++ {
			logSum := mathutil.LogZero
			for j := 1; j <= NumEmittingStates; j++ {
				logSum = mathutil.LogAdd(logSum,
					hmm.TransLog[i][j]+emit[t+1][j-1]+beta[t+1][j])
			}
			beta[t][i] = logSum
		}
	}
}

// TrainPhoneme runs Baum-Welch (EM) on a single phoneme HMM given training sequences.
// sequences[i] is a sequence of feature vectors for one utterance segment.
func TrainPhoneme(hmm *PhonemeHMM, sequences [][][]float64, cfg TrainingConfig) error {
	if len(sequences) == 0 {
		return fmt.Errorf("no training sequences")
	}
	dim := len(sequences[0][0])
	numMix := len(hmm.States[1].GMM.Components)
	prevLL := math.Inf(-1)

	// Find max sequence length for workspace pre-allocation
	maxT := 0
	for _, seq := range sequences {
		if len(seq) > maxT {
			maxT = len(seq)
		}
	}

	// Pre-allocate workspaces (reused across iterations and sequences)
	alpha := mathutil.NewMat(maxT, NumStatesPerPhoneme)
	beta := mathutil.NewMat(maxT, NumStatesPerPhoneme)
	gamma := mathutil.NewMat(maxT, NumStatesPerPhoneme)
	emit := mathutil.NewMat(maxT, NumEmittingStates)

	// Pre-allocate accumulator storage
	transAcc := mathutil.NewMat(NumStatesPerPhoneme, NumStatesPerPhoneme)
	type gmmAcc struct {
		weightAcc []float64   // [numMix] log domain
		meanAcc   [][]float64 // [numMix][dim]
		varAcc    [][]float64 // [numMix][dim]
		totalOcc  float64     // log domain
	}
	stateAcc := make([]gmmAcc, NumEmittingStates)
	for s := range stateAcc {
		stateAcc[s] = gmmAcc{
			weightAcc: make([]float64, numMix),
			meanAcc:   mathutil.NewMat(numMix, dim),
			varAcc:    mathutil.NewMat(numMix, dim),
		}
	}

	for iter := 0; iter < cfg.MaxIterations; iter++ {
		// Reset accumulators (reuse pre-allocated storage)
		mathutil.FillMat(transAcc, mathutil.LogZero)
		for s := range stateAcc {
			mathutil.FillVec(stateAcc[s].weightAcc, mathutil.LogZero)
			mathutil.FillMat(stateAcc[s].meanAcc, 0)
			mathutil.FillMat(stateAcc[s].varAcc, 0)
			stateAcc[s].totalOcc = mathutil.LogZero
		}

		totalLL := 0.0

		for _, obs := range sequences {
			T := len(obs)
			if T == 0 {
				continue
			}

			// Compute emissions once per sequence (avoids redundant GMM evals)
			computeEmissions(hmm, obs, emit[:T])

			// Forward/Backward using pre-allocated workspaces and cached emissions
			forwardWithEmit(hmm, emit[:T], T, alpha[:T])
			backwardWithEmit(hmm, emit[:T], T, beta[:T])

			// Compute P(O|λ) = Σ_j α(T-1,j) * β(T-1,j) including exit.
			ll := mathutil.LogZero
			for j := 1; j <= NumEmittingStates; j++ {
				ll = mathutil.LogAdd(ll, alpha[T-1][j]+beta[T-1][j])
			}
			if ll <= mathutil.LogZero+1 {
				continue
			}
			totalLL += ll

			// Compute gamma[t][j] = log P(q_t=j | O, model)
			for t := 0; t < T; t++ {
				for j := 0; j < NumStatesPerPhoneme; j++ {
					gamma[t][j] = mathutil.LogZero
				}
				for j := 1; j <= NumEmittingStates; j++ {
					gamma[t][j] = alpha[t][j] + beta[t][j] - ll
				}
			}

			// Accumulate transition counts (using cached emissions)
			for t := 0; t < T-1; t++ {
				for i := 1; i <= NumEmittingStates; i++ {
					for j := 1; j <= NumEmittingStates; j++ {
						if hmm.TransLog[i][j] > mathutil.LogZero+1 {
							xi := alpha[t][i] + hmm.TransLog[i][j] +
								emit[t+1][j-1] + beta[t+1][j] - ll
							transAcc[i][j] = mathutil.LogAdd(transAcc[i][j], xi)
						}
					}
				}
			}

			// Accumulate exit transitions at the last frame.
			// At T-1 the model must exit the HMM, so
			// ξ(T-1, i, exit) = α(T-1,i) + a(i,exit) - ll.
			{
				exitJ := NumStatesPerPhoneme - 1
				lastT := T - 1
				for i := 1; i <= NumEmittingStates; i++ {
					if alpha[lastT][i] > mathutil.LogZero+1 && hmm.TransLog[i][exitJ] > mathutil.LogZero+1 {
						xi := alpha[lastT][i] + hmm.TransLog[i][exitJ] - ll
						transAcc[i][exitJ] = mathutil.LogAdd(transAcc[i][exitJ], xi)
					}
				}
			}

			// Accumulate GMM statistics (using cached gmmLP from emit)
			for t := 0; t < T; t++ {
				ot := obs[t]
				gammaT := gamma[t]
				emitT := emit[t]
				for s := 1; s <= NumEmittingStates; s++ {
					if gammaT[s] <= mathutil.LogZero+1 {
						continue
					}
					sIdx := s - 1
					acc := &stateAcc[sIdx]
					gmm := hmm.States[s].GMM
					gmmLP := emitT[sIdx]

					for m := range gmm.Components {
						compLP := gmm.Components[m].LogWeight + gmm.Components[m].LogProb(ot)
						compPost := gammaT[s] + compLP - gmmLP

						acc.weightAcc[m] = mathutil.LogAdd(acc.weightAcc[m], compPost)
						acc.totalOcc = mathutil.LogAdd(acc.totalOcc, compPost)

						postLin := math.Exp(compPost)
						meanRow := acc.meanAcc[m]
						varRow := acc.varAcc[m]
						for d := 0; d < dim; d++ {
							xd := ot[d]
							scaled := postLin * xd
							meanRow[d] += scaled
							varRow[d] += scaled * xd
						}
					}
				}
			}
		}

		// Check convergence
		if iter > 0 && totalLL-prevLL < cfg.ConvergenceThresh {
			break
		}
		prevLL = totalLL

		// Re-estimate transition probabilities
		for i := 1; i <= NumEmittingStates; i++ {
			denom := mathutil.LogZero
			for j := 1; j < NumStatesPerPhoneme; j++ {
				denom = mathutil.LogAdd(denom, transAcc[i][j])
			}
			if denom > mathutil.LogZero+1 {
				for j := 1; j < NumStatesPerPhoneme; j++ {
					if transAcc[i][j] > mathutil.LogZero+1 {
						hmm.TransLog[i][j] = transAcc[i][j] - denom
					} else {
						hmm.TransLog[i][j] = mathutil.LogZero
					}
				}
			}
		}

		// Re-estimate GMM parameters
		for s := 0; s < NumEmittingStates; s++ {
			gmm := hmm.States[s+1].GMM
			acc := stateAcc[s]

			for m := range gmm.Components {
				if acc.weightAcc[m] <= mathutil.LogZero+1 {
					continue
				}
				occLin := math.Exp(acc.weightAcc[m])
				if occLin < 1e-30 {
					continue
				}

				// Update weight
				gmm.Components[m].LogWeight = acc.weightAcc[m] - acc.totalOcc

				// Update mean and variance
				for d := 0; d < dim; d++ {
					gmm.Components[m].Mean[d] = acc.meanAcc[m][d] / occLin
					v := acc.varAcc[m][d]/occLin - gmm.Components[m].Mean[d]*gmm.Components[m].Mean[d]
					if v < cfg.MinVariance {
						v = cfg.MinVariance
					}
					gmm.Components[m].Variance[d] = v
				}
				gmm.Components[m].Precompute()
			}
			// Rebuild SoA cache after parameter update
			gmm.PrecomputeSoA()
		}
	}

	return nil
}
