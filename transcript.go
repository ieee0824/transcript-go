package transcript

import (
	"fmt"
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
	AM      *acoustic.AcousticModel
	LM      *language.NGramModel
	Dict    *lexicon.Dictionary
	FeatCfg feature.Config
	DecCfg  decoder.Config
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
	features, err := feature.Extract(samples, r.FeatCfg)
	if err != nil {
		return nil, fmt.Errorf("extract features: %w", err)
	}
	result := decoder.Decode(features, r.AM, r.LM, r.Dict, r.DecCfg)
	return result, nil
}
