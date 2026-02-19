package transcript

import (
	"fmt"
	"math"
	"os"
	"github.com/ieee0824/transcript-go/acoustic"
	"github.com/ieee0824/transcript-go/audio"
	"github.com/ieee0824/transcript-go/decoder"
	"github.com/ieee0824/transcript-go/feature"
	"github.com/ieee0824/transcript-go/language"
	"github.com/ieee0824/transcript-go/lexicon"
)

// Recognizer is the top-level speech recognizer.
type Recognizer struct {
	AM         *acoustic.AcousticModel
	LM         *language.NGramModel
	Dict       *lexicon.Dictionary
	FeatCfg    feature.Config
	DecCfg     decoder.Config
	OOVLogProb float64 // OOV unigram log10 probability (e.g. -5.0). 0 = disable.
	UseVTLN    bool    // enable VTLN speaker normalization
	dnnPending *acoustic.DNN // set by WithDNN, applied after AM load
}

// Option configures a Recognizer.
type Option func(*Recognizer)

// WithFeatureConfig sets custom MFCC parameters.
func WithFeatureConfig(cfg feature.Config) Option {
	return func(r *Recognizer) {
		r.FeatCfg = cfg
	}
}

// WithDecoderConfig sets custom decoder parameters.
func WithDecoderConfig(cfg decoder.Config) Option {
	return func(r *Recognizer) {
		r.DecCfg = cfg
	}
}

// WithOOVLogProb sets the OOV unigram probability in log10 (e.g. -5.0).
func WithOOVLogProb(log10prob float64) Option {
	return func(r *Recognizer) {
		r.OOVLogProb = log10prob
	}
}

// WithVTLN enables or disables VTLN speaker normalization.
func WithVTLN(enabled bool) Option {
	return func(r *Recognizer) {
		r.UseVTLN = enabled
	}
}

// WithDNN loads a DNN model and attaches it to the acoustic model.
func WithDNN(dnnPath string) Option {
	return func(r *Recognizer) {
		if dnnPath == "" {
			return
		}
		f, err := os.Open(dnnPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "warning: open DNN: %v\n", err)
			return
		}
		defer f.Close()
		dnn, err := acoustic.LoadDNN(f)
		if err != nil {
			fmt.Fprintf(os.Stderr, "warning: load DNN: %v\n", err)
			return
		}
		r.dnnPending = dnn
	}
}

// NewRecognizer creates a Recognizer from model files.
func NewRecognizer(amPath, lmPath, dictPath string, opts ...Option) (*Recognizer, error) {
	r := &Recognizer{
		FeatCfg: feature.DefaultConfig(),
		DecCfg:  decoder.DefaultConfig(),
	}
	for _, opt := range opts {
		opt(r)
	}

	// Load acoustic model
	amFile, err := os.Open(amPath)
	if err != nil {
		return nil, fmt.Errorf("open acoustic model: %w", err)
	}
	defer amFile.Close()
	r.AM, err = acoustic.Load(amFile)
	if err != nil {
		return nil, fmt.Errorf("load acoustic model: %w", err)
	}

	// Load language model
	lmFile, err := os.Open(lmPath)
	if err != nil {
		return nil, fmt.Errorf("open language model: %w", err)
	}
	defer lmFile.Close()
	r.LM, err = language.LoadARPA(lmFile)
	if err != nil {
		return nil, fmt.Errorf("load language model: %w", err)
	}

	// Load dictionary
	r.Dict, err = lexicon.LoadFile(dictPath)
	if err != nil {
		return nil, fmt.Errorf("load dictionary: %w", err)
	}

	// Apply OOV log probability to LM
	if r.OOVLogProb != 0 {
		r.LM.OOVLogProb = r.OOVLogProb * math.Ln10 // convert log10 to natural log
	}

	// Attach DNN to AM if loaded
	if r.dnnPending != nil {
		r.AM.DNN = r.dnnPending
		r.dnnPending = nil
	}

	return r, nil
}

// NewRecognizerFromModels creates a Recognizer from pre-loaded models.
func NewRecognizerFromModels(am *acoustic.AcousticModel, lm *language.NGramModel, dict *lexicon.Dictionary, opts ...Option) *Recognizer {
	r := &Recognizer{
		AM:      am,
		LM:      lm,
		Dict:    dict,
		FeatCfg: feature.DefaultConfig(),
		DecCfg:  decoder.DefaultConfig(),
	}
	for _, opt := range opts {
		opt(r)
	}
	return r
}

// RecognizeFile runs recognition on a WAV file and returns the result.
func (r *Recognizer) RecognizeFile(wavPath string) (*decoder.Result, error) {
	samples, _, err := audio.ReadWAVFile(wavPath)
	if err != nil {
		return nil, fmt.Errorf("read WAV: %w", err)
	}
	return r.RecognizeSamples(samples)
}

// RecognizeSamples runs recognition on raw audio samples.
func (r *Recognizer) RecognizeSamples(samples []float64) (*decoder.Result, error) {
	var features [][]float64
	var err error
	if r.UseVTLN {
		scorer := func(feats [][]float64) float64 {
			return acoustic.FrameLikelihood(r.AM, feats)
		}
		features, _, err = feature.ExtractWithVTLN(samples, r.FeatCfg, scorer)
	} else {
		features, err = feature.Extract(samples, r.FeatCfg)
	}
	if err != nil {
		return nil, fmt.Errorf("extract features: %w", err)
	}
	result := decoder.Decode(features, r.AM, r.LM, r.Dict, r.DecCfg)
	return result, nil
}
