package acoustic

import (
	"math"
	"math/rand"
	"github.com/ieee0824/transcript-go/internal/mathutil"
	"github.com/ieee0824/transcript-go/internal/simd"
)

// Gaussian represents a single multivariate Gaussian component with diagonal covariance.
type Gaussian struct {
	Mean     []float64 // [dim]
	Variance []float64 // [dim] diagonal covariance
	LogWeight float64  // log mixture weight

	// Pre-computed values
	logNormConst float64
	invVariance  []float64 // [dim] 1/Variance, precomputed to avoid division in hot loop
}

// Precompute recalculates cached normalization constants and inverse variances.
// Must be called after updating Mean, Variance, or LogWeight.
func (g *Gaussian) Precompute() {
	dim := len(g.Mean)
	g.logNormConst = float64(dim)/2.0*math.Log(2*math.Pi) + 0.5*sumLog(g.Variance)
	g.invVariance = make([]float64, dim)
	for i := range g.Variance {
		g.invVariance[i] = 1.0 / g.Variance[i]
	}
}

// LogProb computes the log probability of observation x under this Gaussian.
func (g *Gaussian) LogProb(x []float64) float64 {
	maha := simd.MahalanobisAccum(x, g.Mean, g.invVariance)
	return -0.5*maha - g.logNormConst
}

func sumLog(v []float64) float64 {
	s := 0.0
	for _, x := range v {
		s += math.Log(x)
	}
	return s
}

// GMM is a Gaussian Mixture Model with diagonal covariance.
type GMM struct {
	Components []Gaussian
	Dim        int

	// SoA (Struct of Arrays) cache for fast LogProb.
	// Built by Precompute(). All component data packed contiguously.
	soaMean    []float64 // [k*dim] - means packed contiguously
	soaInvVar  []float64 // [k*dim] - inverse variances packed contiguously
	soaConst   []float64 // [k] - logWeight - logNormConst per component
}

// PrecomputeSoA builds the SoA cache for fast LogProb. Call after all components are set.
func (g *GMM) PrecomputeSoA() {
	k := len(g.Components)
	dim := g.Dim
	g.soaMean = make([]float64, k*dim)
	g.soaInvVar = make([]float64, k*dim)
	g.soaConst = make([]float64, k)
	for i := range g.Components {
		g.Components[i].Precompute()
		off := i * dim
		copy(g.soaMean[off:off+dim], g.Components[i].Mean)
		copy(g.soaInvVar[off:off+dim], g.Components[i].invVariance)
		g.soaConst[i] = g.Components[i].LogWeight - g.Components[i].logNormConst
	}
}

// NewGMM creates a GMM with k components of dimension dim, initialized randomly.
func NewGMM(k, dim int) *GMM {
	g := &GMM{
		Components: make([]Gaussian, k),
		Dim:        dim,
	}
	logW := -math.Log(float64(k))
	for i := range g.Components {
		mean := make([]float64, dim)
		variance := make([]float64, dim)
		for d := 0; d < dim; d++ {
			mean[d] = rand.NormFloat64()
			variance[d] = 1.0
		}
		g.Components[i] = Gaussian{
			Mean:      mean,
			Variance:  variance,
			LogWeight: logW,
		}
		g.Components[i].Precompute()
	}
	g.PrecomputeSoA()
	return g
}

// NewGMMWithParams creates a GMM from given parameters.
func NewGMMWithParams(means, variances [][]float64, logWeights []float64) *GMM {
	k := len(means)
	dim := len(means[0])
	g := &GMM{
		Components: make([]Gaussian, k),
		Dim:        dim,
	}
	for i := range g.Components {
		mean := make([]float64, dim)
		variance := make([]float64, dim)
		copy(mean, means[i])
		copy(variance, variances[i])
		g.Components[i] = Gaussian{
			Mean:      mean,
			Variance:  variance,
			LogWeight: logWeights[i],
		}
		g.Components[i].Precompute()
	}
	g.PrecomputeSoA()
	return g
}

// LogProb computes log P(x | this GMM) = log sum_k w_k * N(x; μ_k, σ_k).
// Uses SoA layout for cache-friendly access when available.
func (g *GMM) LogProb(x []float64) float64 {
	if g.soaMean != nil {
		return g.logProbSoA(x)
	}
	logSum := mathutil.LogZero
	for i := range g.Components {
		lp := g.Components[i].LogWeight + g.Components[i].LogProb(x)
		logSum = mathutil.LogAdd(logSum, lp)
	}
	return logSum
}

// logProbSoA uses the packed SoA layout for better cache locality.
func (g *GMM) logProbSoA(x []float64) float64 {
	k := len(g.Components)
	dim := g.Dim
	logSum := mathutil.LogZero
	mean := g.soaMean
	invVar := g.soaInvVar

	for c := 0; c < k; c++ {
		off := c * dim
		cm := mean[off : off+dim]
		cv := invVar[off : off+dim]
		maha := simd.MahalanobisAccum(x, cm, cv)
		lp := g.soaConst[c] - 0.5*maha
		logSum = mathutil.LogAdd(logSum, lp)
	}
	return logSum
}

// LogProbBatch computes LogProb for multiple observations, writing results into dst.
// Keeps SoA data in cache across frames for better throughput.
func (g *GMM) LogProbBatch(xs [][]float64, dst []float64) {
	if g.soaMean == nil {
		for i, x := range xs {
			dst[i] = g.LogProb(x)
		}
		return
	}
	k := len(g.Components)
	dim := g.Dim
	mean := g.soaMean
	invVar := g.soaInvVar
	soaConst := g.soaConst

	for fi, x := range xs {
		logSum := mathutil.LogZero
		for c := 0; c < k; c++ {
			off := c * dim
			cm := mean[off : off+dim]
			cv := invVar[off : off+dim]
			maha := simd.MahalanobisAccum(x, cm, cv)
			lp := soaConst[c] - 0.5*maha
			logSum = mathutil.LogAdd(logSum, lp)
		}
		dst[fi] = logSum
	}
}
