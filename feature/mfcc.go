package feature

import (
	"fmt"
	"math"
)

// Config holds all MFCC extraction parameters.
type Config struct {
	SampleRate    int
	FrameLenMs    float64 // frame length in milliseconds
	FrameShiftMs  float64 // frame shift in milliseconds
	PreEmphCoeff  float64
	NumMelFilters int
	NumCepstra    int
	LowFreq       float64
	HighFreq      float64
	FFTSize       int
	UseDelta      bool
	UseDeltaDelta bool
	CepLifter     int
	UseCMN        bool    // cepstral mean normalization
	Alpha         float64 // VTLN warp factor (1.0 = no warping)
}

// DefaultConfig returns the standard MFCC configuration.
func DefaultConfig() Config {
	return Config{
		SampleRate:    16000,
		FrameLenMs:    25.0,
		FrameShiftMs:  10.0,
		PreEmphCoeff:  0.97,
		NumMelFilters: 26,
		NumCepstra:    13,
		LowFreq:       0,
		HighFreq:      8000,
		FFTSize:       512,
		UseDelta:      true,
		UseDeltaDelta: true,
		CepLifter:     22,
		UseCMN:        true,
		Alpha:         1.0,
	}
}

// FeatureDim returns the total feature vector dimension.
func (c Config) FeatureDim() int {
	d := c.NumCepstra
	if c.UseDelta {
		d += c.NumCepstra
	}
	if c.UseDeltaDelta {
		d += c.NumCepstra
	}
	return d
}

// Extract computes MFCC features from raw audio samples.
// Returns a matrix of shape [numFrames][numFeatures].
func Extract(samples []float64, cfg Config) ([][]float64, error) {
	if len(samples) == 0 {
		return nil, fmt.Errorf("empty samples")
	}

	frameLen := int(cfg.FrameLenMs * float64(cfg.SampleRate) / 1000.0)
	frameShift := int(cfg.FrameShiftMs * float64(cfg.SampleRate) / 1000.0)

	// 1. Pre-emphasis
	emphasized := PreEmphasize(samples, cfg.PreEmphCoeff)

	// 2. Framing
	frames := Frame(emphasized, frameLen, frameShift)
	if len(frames) == 0 {
		return nil, fmt.Errorf("audio too short for a single frame")
	}

	// 3. Build reusable workspace (once)
	melFB := NewMelFilterbank(cfg.NumMelFilters, cfg.FFTSize, cfg.SampleRate, cfg.LowFreq, cfg.HighFreq, cfg.Alpha)
	fftWS := newFFTWorkspace(cfg.FFTSize)
	dctTbl := newDCTTable(cfg.NumCepstra, cfg.NumMelFilters)
	melBuf := make([]float64, cfg.NumMelFilters)

	var liftTbl *lifterTable
	if cfg.CepLifter > 0 {
		liftTbl = newLifterTable(cfg.NumCepstra, cfg.CepLifter)
	}

	// 4. For each frame: window+FFT -> power spectrum -> Mel -> DCT -> lifter
	nFrames := len(frames)
	mfccs := make([][]float64, nFrames)
	cepAll := make([]float64, nFrames*cfg.NumCepstra)
	hammWin := getHammingWindow(frameLen)
	for i, frame := range frames {
		fftWS.computePowerSpectrum(frame, hammWin)
		melFB.applyInto(fftWS.power, melBuf)
		cepstra := cepAll[i*cfg.NumCepstra : (i+1)*cfg.NumCepstra]
		dctTbl.applyInto(melBuf, cepstra)
		if liftTbl != nil {
			liftTbl.apply(cepstra)
		}
		mfccs[i] = cepstra
	}

	// 4.5. Cepstral mean normalization (before delta)
	if cfg.UseCMN {
		ApplyCMN(mfccs)
	}

	// 5. Append deltas
	if cfg.UseDelta && cfg.UseDeltaDelta {
		mfccs = AppendDeltas(mfccs)
	} else if cfg.UseDelta {
		d1 := Delta(mfccs, 2)
		for t := range mfccs {
			mfccs[t] = append(mfccs[t], d1[t]...)
		}
	}

	return mfccs, nil
}

// ExtractPowerSpectra computes per-frame FFT power spectra from raw audio samples.
// The expensive FFT is done once; callers can apply different mel filterbanks
// (with varying VTLN alpha) to the same spectra via FeaturesFromSpectra.
func ExtractPowerSpectra(samples []float64, cfg Config) ([][]float64, error) {
	if len(samples) == 0 {
		return nil, fmt.Errorf("empty samples")
	}

	frameLen := int(cfg.FrameLenMs * float64(cfg.SampleRate) / 1000.0)
	frameShift := int(cfg.FrameShiftMs * float64(cfg.SampleRate) / 1000.0)

	emphasized := PreEmphasize(samples, cfg.PreEmphCoeff)
	frames := Frame(emphasized, frameLen, frameShift)
	if len(frames) == 0 {
		return nil, fmt.Errorf("audio too short for a single frame")
	}

	fftWS := newFFTWorkspace(cfg.FFTSize)
	nBins := cfg.FFTSize/2 + 1
	hammWin := getHammingWindow(frameLen)

	spectra := make([][]float64, len(frames))
	for i, frame := range frames {
		fftWS.computePowerSpectrum(frame, hammWin)
		spectra[i] = make([]float64, nBins)
		copy(spectra[i], fftWS.power[:nBins])
	}
	return spectra, nil
}

// FeaturesFromSpectra computes MFCC features from pre-computed power spectra.
// Uses cfg.Alpha for VTLN warping of the mel filterbank.
func FeaturesFromSpectra(spectra [][]float64, cfg Config) [][]float64 {
	nFrames := len(spectra)
	if nFrames == 0 {
		return nil
	}

	melFB := NewMelFilterbank(cfg.NumMelFilters, cfg.FFTSize, cfg.SampleRate,
		cfg.LowFreq, cfg.HighFreq, cfg.Alpha)
	dctTbl := newDCTTable(cfg.NumCepstra, cfg.NumMelFilters)
	melBuf := make([]float64, cfg.NumMelFilters)

	var liftTbl *lifterTable
	if cfg.CepLifter > 0 {
		liftTbl = newLifterTable(cfg.NumCepstra, cfg.CepLifter)
	}

	mfccs := make([][]float64, nFrames)
	cepAll := make([]float64, nFrames*cfg.NumCepstra)
	for i, ps := range spectra {
		melFB.applyInto(ps, melBuf)
		cepstra := cepAll[i*cfg.NumCepstra : (i+1)*cfg.NumCepstra]
		dctTbl.applyInto(melBuf, cepstra)
		if liftTbl != nil {
			liftTbl.apply(cepstra)
		}
		mfccs[i] = cepstra
	}

	if cfg.UseCMN {
		ApplyCMN(mfccs)
	}

	if cfg.UseDelta && cfg.UseDeltaDelta {
		mfccs = AppendDeltas(mfccs)
	} else if cfg.UseDelta {
		d1 := Delta(mfccs, 2)
		for t := range mfccs {
			mfccs[t] = append(mfccs[t], d1[t]...)
		}
	}
	return mfccs
}

// ExtractWithVTLN extracts features with VTLN normalization via grid search.
// It computes FFT power spectra once, then for each candidate alpha value
// applies a warped mel filterbank and scores with the provided function.
// Returns the best features, best alpha, and any error.
func ExtractWithVTLN(samples []float64, cfg Config, scorer func([][]float64) float64) ([][]float64, float64, error) {
	spectra, err := ExtractPowerSpectra(samples, cfg)
	if err != nil {
		return nil, 1.0, err
	}

	bestScore := math.Inf(-1)
	bestAlpha := 1.0
	var bestFeatures [][]float64

	for a := 0.82; a <= 1.201; a += 0.02 {
		alpha := math.Round(a*100) / 100
		tryCfg := cfg
		tryCfg.Alpha = alpha
		features := FeaturesFromSpectra(spectra, tryCfg)
		score := scorer(features)
		if score > bestScore {
			bestScore = score
			bestAlpha = alpha
			bestFeatures = features
		}
	}

	return bestFeatures, bestAlpha, nil
}
