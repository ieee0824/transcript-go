package acoustic

import (
	"encoding/gob"
	"io"
	"math"
	"math/rand"

	"github.com/ieee0824/transcript-go/internal/blas"
)

// DNN is a feedforward neural network for HMM state classification.
// Architecture: input → hidden1 (ReLU) → hidden2 (ReLU) → output (log-softmax).
// Used in DNN-HMM hybrid: replaces GMM emission scoring while keeping HMM transitions.
type DNN struct {
	InputDim  int // context window: (2*ContextLen+1) * FeatureDim
	HiddenDim int
	OutputDim int // NumPhonemes * NumEmittingStates (87)
	ContextLen int // frames on each side (e.g. 5 → 11-frame window)

	// Weights (row-major flat slices for BLAS)
	W1 []float64 // [HiddenDim × InputDim]
	B1 []float64 // [HiddenDim]
	W2 []float64 // [HiddenDim × HiddenDim]
	B2 []float64 // [HiddenDim]
	W3 []float64 // [OutputDim × HiddenDim]
	B3 []float64 // [OutputDim]

	// Log prior P(state) from training label histogram, for Bayes conversion
	LogPrior []float64 // [OutputDim]

	// Ordered phoneme list for class index mapping
	PhonemeList []Phoneme // len = number of phonemes used
}

// NewDNN creates a DNN with Xavier-initialized weights.
func NewDNN(featureDim, hiddenDim, contextLen int) *DNN {
	phonemes := AllPhonemes()
	outputDim := len(phonemes) * NumEmittingStates
	inputDim := (2*contextLen + 1) * featureDim

	d := &DNN{
		InputDim:    inputDim,
		HiddenDim:   hiddenDim,
		OutputDim:   outputDim,
		ContextLen:  contextLen,
		W1:          make([]float64, hiddenDim*inputDim),
		B1:          make([]float64, hiddenDim),
		W2:          make([]float64, hiddenDim*hiddenDim),
		B2:          make([]float64, hiddenDim),
		W3:          make([]float64, outputDim*hiddenDim),
		B3:          make([]float64, outputDim),
		LogPrior:    make([]float64, outputDim),
		PhonemeList: phonemes,
	}

	// Xavier initialization: N(0, sqrt(2/(fan_in+fan_out)))
	xavierInit(d.W1, inputDim, hiddenDim)
	xavierInit(d.W2, hiddenDim, hiddenDim)
	xavierInit(d.W3, hiddenDim, outputDim)

	return d
}

func xavierInit(w []float64, fanIn, fanOut int) {
	scale := math.Sqrt(2.0 / float64(fanIn+fanOut))
	for i := range w {
		w[i] = rand.NormFloat64() * scale
	}
}

// StateClassIndex returns the DNN output class index for a phoneme and emitting state (1-based).
// Returns -1 if the phoneme is not found.
func (d *DNN) StateClassIndex(ph Phoneme, stateIdx int) int {
	for i, p := range d.PhonemeList {
		if p == ph {
			return i*NumEmittingStates + (stateIdx - 1)
		}
	}
	return -1
}

// Forward computes log-softmax outputs for a batch of input vectors using BLAS.
// input: flat [batchSize × InputDim] row-major
// output: flat [batchSize × OutputDim] log-softmax values
// work buffers h1, h2 must be [batchSize × HiddenDim].
func (d *DNN) Forward(input []float64, batchSize int, h1, h2, output []float64) {
	H := d.HiddenDim
	O := d.OutputDim
	I := d.InputDim

	// Layer 1: h1 = input @ W1^T + B1, then ReLU
	// h1[B×H] = input[B×I] × W1^T[I×H]
	blas.Dgemm(false, true, batchSize, H, I,
		1.0, input, I, d.W1, I, 0.0, h1, H)
	addBiasReLU(h1, d.B1, batchSize, H)

	// Layer 2: h2 = h1 @ W2^T + B2, then ReLU
	blas.Dgemm(false, true, batchSize, H, H,
		1.0, h1, H, d.W2, H, 0.0, h2, H)
	addBiasReLU(h2, d.B2, batchSize, H)

	// Layer 3: output = h2 @ W3^T + B3, then log-softmax
	blas.Dgemm(false, true, batchSize, O, H,
		1.0, h2, H, d.W3, H, 0.0, output, O)
	addBiasLogSoftmax(output, d.B3, batchSize, O)
}

// addBiasReLU adds bias and applies ReLU in place.
func addBiasReLU(z []float64, bias []float64, rows, cols int) {
	for i := 0; i < rows; i++ {
		off := i * cols
		for j := 0; j < cols; j++ {
			v := z[off+j] + bias[j]
			if v < 0 {
				v = 0
			}
			z[off+j] = v
		}
	}
}

// addBiasLogSoftmax adds bias and applies log-softmax per row.
func addBiasLogSoftmax(z []float64, bias []float64, rows, cols int) {
	for i := 0; i < rows; i++ {
		off := i * cols
		// Add bias and find max for numerical stability
		maxVal := math.Inf(-1)
		for j := 0; j < cols; j++ {
			z[off+j] += bias[j]
			if z[off+j] > maxVal {
				maxVal = z[off+j]
			}
		}
		// log-softmax: log(exp(z_j - max) / sum(exp(z_k - max)))
		sumExp := 0.0
		for j := 0; j < cols; j++ {
			sumExp += math.Exp(z[off+j] - maxVal)
		}
		logSumExp := maxVal + math.Log(sumExp)
		for j := 0; j < cols; j++ {
			z[off+j] -= logSumExp
		}
	}
}

// ForwardFrames computes DNN log-posteriors for all T frames.
// Handles context window construction with edge replication padding.
// Returns [T][OutputDim] log-posteriors.
func (d *DNN) ForwardFrames(features [][]float64) [][]float64 {
	T := len(features)
	if T == 0 {
		return nil
	}

	// Build context-windowed input matrix: [T × InputDim] flat
	input := make([]float64, T*d.InputDim)
	featDim := len(features[0])
	winSize := 2*d.ContextLen + 1

	for t := 0; t < T; t++ {
		off := t * d.InputDim
		for w := 0; w < winSize; w++ {
			srcT := t - d.ContextLen + w
			if srcT < 0 {
				srcT = 0
			} else if srcT >= T {
				srcT = T - 1
			}
			copy(input[off+w*featDim:off+(w+1)*featDim], features[srcT])
		}
	}

	// Allocate work buffers and output
	h1 := make([]float64, T*d.HiddenDim)
	h2 := make([]float64, T*d.HiddenDim)
	outFlat := make([]float64, T*d.OutputDim)

	d.Forward(input, T, h1, h2, outFlat)

	// Reshape to [][]float64
	result := make([][]float64, T)
	for t := 0; t < T; t++ {
		result[t] = outFlat[t*d.OutputDim : (t+1)*d.OutputDim]
	}
	return result
}

// SubtractPrior converts log-posteriors to pseudo-log-likelihoods in place.
// logLike[c] = logPost[c] - logPrior[c]
func (d *DNN) SubtractPrior(logPost []float64) {
	for i, lp := range d.LogPrior {
		logPost[i] -= lp
	}
}

// serialized DNN for gob encoding
type serializedDNN struct {
	InputDim    int
	HiddenDim   int
	OutputDim   int
	ContextLen  int
	W1, B1      []float64
	W2, B2      []float64
	W3, B3      []float64
	LogPrior    []float64
	PhonemeList []string
}

// Save serializes the DNN to a writer using gob encoding.
func (d *DNN) Save(w io.Writer) error {
	sd := serializedDNN{
		InputDim:   d.InputDim,
		HiddenDim:  d.HiddenDim,
		OutputDim:  d.OutputDim,
		ContextLen: d.ContextLen,
		W1:         d.W1, B1: d.B1,
		W2:         d.W2, B2: d.B2,
		W3:         d.W3, B3: d.B3,
		LogPrior:   d.LogPrior,
	}
	sd.PhonemeList = make([]string, len(d.PhonemeList))
	for i, p := range d.PhonemeList {
		sd.PhonemeList[i] = string(p)
	}
	return gob.NewEncoder(w).Encode(sd)
}

// LoadDNN deserializes a DNN from a reader.
func LoadDNN(r io.Reader) (*DNN, error) {
	var sd serializedDNN
	if err := gob.NewDecoder(r).Decode(&sd); err != nil {
		return nil, err
	}
	d := &DNN{
		InputDim:   sd.InputDim,
		HiddenDim:  sd.HiddenDim,
		OutputDim:  sd.OutputDim,
		ContextLen: sd.ContextLen,
		W1:         sd.W1, B1: sd.B1,
		W2:         sd.W2, B2: sd.B2,
		W3:         sd.W3, B3: sd.B3,
		LogPrior:   sd.LogPrior,
	}
	d.PhonemeList = make([]Phoneme, len(sd.PhonemeList))
	for i, s := range sd.PhonemeList {
		d.PhonemeList[i] = Phoneme(s)
	}
	return d, nil
}
