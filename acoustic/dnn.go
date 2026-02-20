package acoustic

import (
	"bytes"
	"encoding/gob"
	"io"
	"math"
	"math/rand"

	"github.com/ieee0824/transcript-go/internal/blas"
)

// DNNLayer holds weights and biases for a single fully-connected layer.
// W is [OutDim × InDim] row-major, B is [OutDim].
type DNNLayer struct {
	W      []float64
	B      []float64
	InDim  int
	OutDim int
}

// DNN is a feedforward neural network for HMM state classification.
// Architecture: input → hidden1 (ReLU) → ... → hiddenN (ReLU) → output (log-softmax).
// Layers[0..N-2] are hidden layers with ReLU, Layers[N-1] is the output layer with log-softmax.
// Used in DNN-HMM hybrid: replaces GMM emission scoring while keeping HMM transitions.
type DNN struct {
	Layers      []DNNLayer // hidden layers + output layer
	InputDim    int        // = Layers[0].InDim (cached for compatibility)
	HiddenDim   int        // = Layers[0].OutDim (cached for display)
	OutputDim   int        // = Layers[N-1].OutDim (cached)
	ContextLen  int        // frames on each side (e.g. 5 → 11-frame window)
	DropoutRate float64    // inverted dropout rate for hidden layers (0 = disabled)

	// Log prior P(state) from training label histogram, for Bayes conversion
	LogPrior []float64 // [OutputDim]

	// Ordered phoneme list for class index mapping
	PhonemeList []Phoneme // len = number of phonemes used
}

// NewDNN creates a DNN with Xavier-initialized weights.
// numHiddenLayers specifies the number of hidden layers (all with hiddenDim units).
// dropoutRate is the inverted dropout rate for hidden layers (0 = disabled).
func NewDNN(featureDim, hiddenDim, contextLen, numHiddenLayers int, dropoutRate float64) *DNN {
	phonemes := AllPhonemes()
	outputDim := len(phonemes) * NumEmittingStates
	inputDim := (2*contextLen + 1) * featureDim

	layers := make([]DNNLayer, numHiddenLayers+1)
	prevDim := inputDim
	for i := 0; i < numHiddenLayers; i++ {
		layers[i] = DNNLayer{
			W:      make([]float64, hiddenDim*prevDim),
			B:      make([]float64, hiddenDim),
			InDim:  prevDim,
			OutDim: hiddenDim,
		}
		xavierInit(layers[i].W, prevDim, hiddenDim)
		prevDim = hiddenDim
	}
	// Output layer
	layers[numHiddenLayers] = DNNLayer{
		W:      make([]float64, outputDim*prevDim),
		B:      make([]float64, outputDim),
		InDim:  prevDim,
		OutDim: outputDim,
	}
	xavierInit(layers[numHiddenLayers].W, prevDim, outputDim)

	return &DNN{
		Layers:      layers,
		InputDim:    inputDim,
		HiddenDim:   hiddenDim,
		OutputDim:   outputDim,
		ContextLen:  contextLen,
		DropoutRate: dropoutRate,
		LogPrior:    make([]float64, outputDim),
		PhonemeList: phonemes,
	}
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
// activations: per-hidden-layer buffers, each [batchSize × layer.OutDim]
// output: flat [batchSize × OutputDim] log-softmax values
// Dropout is NOT applied (inference path).
func (d *DNN) Forward(input []float64, batchSize int, activations [][]float64, output []float64) {
	nLayers := len(d.Layers)
	prevAct := input
	prevDim := d.InputDim

	for i := range d.Layers {
		layer := &d.Layers[i]
		var dst []float64
		if i < nLayers-1 {
			dst = activations[i]
		} else {
			dst = output
		}

		blas.Dgemm(false, true, batchSize, layer.OutDim, prevDim,
			1.0, prevAct, prevDim, layer.W, prevDim, 0.0, dst, layer.OutDim)

		if i < nLayers-1 {
			addBiasReLU(dst, layer.B, batchSize, layer.OutDim)
		} else {
			addBiasLogSoftmax(dst, layer.B, batchSize, layer.OutDim)
		}

		prevAct = dst
		prevDim = layer.OutDim
	}
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

	// Allocate per-hidden-layer buffers
	nHidden := len(d.Layers) - 1
	activations := make([][]float64, nHidden)
	for i := 0; i < nHidden; i++ {
		activations[i] = make([]float64, T*d.Layers[i].OutDim)
	}
	outFlat := make([]float64, T*d.OutputDim)

	d.Forward(input, T, activations, outFlat)

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

// --- Serialization ---

// V2 serialized format (variable layers)
type serializedDNNLayer struct {
	W      []float64
	B      []float64
	InDim  int
	OutDim int
}

type serializedDNNV2 struct {
	Version     int // = 2
	ContextLen  int
	DropoutRate float64
	Layers      []serializedDNNLayer
	LogPrior    []float64
	PhonemeList []string
}

// V1 serialized format (legacy 3-layer)
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

// Save serializes the DNN to a writer using gob encoding (V2 format).
func (d *DNN) Save(w io.Writer) error {
	sd := serializedDNNV2{
		Version:    2,
		ContextLen: d.ContextLen,
		Layers:     make([]serializedDNNLayer, len(d.Layers)),
		LogPrior:   d.LogPrior,
	}
	for i, l := range d.Layers {
		sd.Layers[i] = serializedDNNLayer{W: l.W, B: l.B, InDim: l.InDim, OutDim: l.OutDim}
	}
	sd.PhonemeList = make([]string, len(d.PhonemeList))
	for i, p := range d.PhonemeList {
		sd.PhonemeList[i] = string(p)
	}
	return gob.NewEncoder(w).Encode(sd)
}

// LoadDNN deserializes a DNN from a reader. Supports both V2 and legacy V1 formats.
func LoadDNN(r io.Reader) (*DNN, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, err
	}

	// Try V2 format
	var v2 serializedDNNV2
	if err := gob.NewDecoder(bytes.NewReader(data)).Decode(&v2); err == nil && v2.Version == 2 && len(v2.Layers) > 0 {
		return dnnFromV2(&v2), nil
	}

	// Fall back to V1 (legacy 3-layer format)
	var v1 serializedDNN
	if err := gob.NewDecoder(bytes.NewReader(data)).Decode(&v1); err != nil {
		return nil, err
	}
	return dnnFromV1(&v1), nil
}

func dnnFromV2(sd *serializedDNNV2) *DNN {
	layers := make([]DNNLayer, len(sd.Layers))
	for i, sl := range sd.Layers {
		layers[i] = DNNLayer{W: sl.W, B: sl.B, InDim: sl.InDim, OutDim: sl.OutDim}
	}
	nLayers := len(layers)
	d := &DNN{
		Layers:     layers,
		InputDim:   layers[0].InDim,
		HiddenDim:  layers[0].OutDim,
		OutputDim:  layers[nLayers-1].OutDim,
		ContextLen: sd.ContextLen,
		LogPrior:   sd.LogPrior,
	}
	d.PhonemeList = make([]Phoneme, len(sd.PhonemeList))
	for i, s := range sd.PhonemeList {
		d.PhonemeList[i] = Phoneme(s)
	}
	return d
}

func dnnFromV1(sd *serializedDNN) *DNN {
	d := &DNN{
		Layers: []DNNLayer{
			{W: sd.W1, B: sd.B1, InDim: sd.InputDim, OutDim: sd.HiddenDim},
			{W: sd.W2, B: sd.B2, InDim: sd.HiddenDim, OutDim: sd.HiddenDim},
			{W: sd.W3, B: sd.B3, InDim: sd.HiddenDim, OutDim: sd.OutputDim},
		},
		InputDim:   sd.InputDim,
		HiddenDim:  sd.HiddenDim,
		OutputDim:  sd.OutputDim,
		ContextLen: sd.ContextLen,
		LogPrior:   sd.LogPrior,
	}
	d.PhonemeList = make([]Phoneme, len(sd.PhonemeList))
	for i, s := range sd.PhonemeList {
		d.PhonemeList[i] = Phoneme(s)
	}
	return d
}
