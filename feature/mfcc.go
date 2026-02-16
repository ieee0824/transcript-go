package feature

import "fmt"

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
	melFB := NewMelFilterbank(cfg.NumMelFilters, cfg.FFTSize, cfg.SampleRate, cfg.LowFreq, cfg.HighFreq)
	fftWS := newFFTWorkspace(cfg.FFTSize)
	dctTbl := newDCTTable(cfg.NumCepstra, cfg.NumMelFilters)
	melBuf := make([]float64, cfg.NumMelFilters)

	var liftTbl *lifterTable
	if cfg.CepLifter > 0 {
		liftTbl = newLifterTable(cfg.NumCepstra, cfg.CepLifter)
	}

	// 4. For each frame: window -> FFT -> power spectrum -> Mel -> DCT -> lifter
	mfccs := make([][]float64, len(frames))
	for i, frame := range frames {
		HammingWindow(frame)
		fftWS.computePowerSpectrum(frame)
		melFB.applyInto(fftWS.power, melBuf)
		cepstra := make([]float64, cfg.NumCepstra)
		dctTbl.applyInto(melBuf, cepstra)
		if liftTbl != nil {
			liftTbl.apply(cepstra)
		}
		mfccs[i] = cepstra
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
