package acoustic

import (
	"math"

	"github.com/ieee0824/transcript-go/internal/blas"
	"github.com/ieee0824/transcript-go/internal/mathutil"
)

// BatchWorkspace holds pre-allocated buffers for LogProbBatchMat.
type BatchWorkspace struct {
	Xsq   []float64 // T * D
	Term1 []float64 // T * maxK
	Term2 []float64 // T * maxK
	LP    []float64 // T * maxK (for logsumexp)
}

// NewBatchWorkspace creates a workspace for T frames, D dimensions, maxK mixture components.
func NewBatchWorkspace(T, D, maxK int) *BatchWorkspace {
	return &BatchWorkspace{
		Xsq:   make([]float64, T*D),
		Term1: make([]float64, T*maxK),
		Term2: make([]float64, T*maxK),
		LP:    make([]float64, T*maxK),
	}
}

// EnsureBatchWorkspace grows workspace buffers if needed.
func (ws *BatchWorkspace) EnsureBatchWorkspace(T, D, maxK int) {
	if cap(ws.Xsq) < T*D {
		ws.Xsq = make([]float64, T*D)
	} else {
		ws.Xsq = ws.Xsq[:T*D]
	}
	if cap(ws.Term1) < T*maxK {
		ws.Term1 = make([]float64, T*maxK)
	} else {
		ws.Term1 = ws.Term1[:T*maxK]
	}
	if cap(ws.Term2) < T*maxK {
		ws.Term2 = make([]float64, T*maxK)
	} else {
		ws.Term2 = ws.Term2[:T*maxK]
	}
	if cap(ws.LP) < T*maxK {
		ws.LP = make([]float64, T*maxK)
	} else {
		ws.LP = ws.LP[:T*maxK]
	}
}

// LogProbBatchMat computes log P(x_t | GMM) for T frames using BLAS matrix multiply.
// xs is a flat [T*D] array of feature vectors (row-major).
// dst is [T] output log-probabilities.
//
// Math:
//
//	maha(x,μ,invVar) = Σ(x²·invVar) - 2·Σ(x·μ·invVar) + Σ(μ²·invVar)
//	term1 = X² @ invVar^T   (T×D) × (K×D)^T → (T×K)
//	term2 = X  @ meanInvVar^T  (T×D) × (K×D)^T → (T×K)
//	lp[t,k] = -0.5*term1[t,k] + term2[t,k] + bias[k]
//	dst[t] = logsumexp_k(lp[t,k])
func (g *GMM) LogProbBatchMat(xs []float64, T, D int, dst []float64, ws *BatchWorkspace) {
	K := len(g.Components)
	ws.EnsureBatchWorkspace(T, D, K)

	// 1. xsq = xs .^ 2
	xsq := ws.Xsq
	for i, v := range xs[:T*D] {
		xsq[i] = v * v
	}

	term1 := ws.Term1
	term2 := ws.Term2

	// 2. term1 = Xsq @ invVar^T  [T×K]
	blas.Dgemm(false, true, T, K, D, 1.0, xsq, D, g.soaInvVar, D, 0.0, term1, K)

	// 3. term2 = X @ meanInvVar^T  [T×K]
	blas.Dgemm(false, true, T, K, D, 1.0, xs, D, g.soaMeanInvVar, D, 0.0, term2, K)

	// 4. lp[t,k] = -0.5*term1 + term2 + bias[k], then logsumexp over k
	bias := g.soaBias
	for t := 0; t < T; t++ {
		rowOff := t * K
		maxLP := -math.MaxFloat64
		for c := 0; c < K; c++ {
			lp := -0.5*term1[rowOff+c] + term2[rowOff+c] + bias[c]
			ws.LP[rowOff+c] = lp
			if lp > maxLP {
				maxLP = lp
			}
		}
		// logsumexp
		if maxLP <= mathutil.LogZero {
			dst[t] = mathutil.LogZero
			continue
		}
		sum := 0.0
		for c := 0; c < K; c++ {
			d := ws.LP[rowOff+c] - maxLP
			if d > -36.0 {
				sum += math.Exp(d)
			}
		}
		dst[t] = maxLP + math.Log(sum)
	}
}
