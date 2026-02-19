package feature

import (
	"math"
	"math/cmplx"
	"testing"
)

func TestPreEmphasize(t *testing.T) {
	samples := []float64{1.0, 2.0, 3.0, 4.0}
	out := PreEmphasize(samples, 0.97)
	if out[0] != 1.0 {
		t.Errorf("out[0] = %f, want 1.0", out[0])
	}
	// out[1] = 2.0 - 0.97*1.0 = 1.03
	if math.Abs(out[1]-1.03) > 1e-10 {
		t.Errorf("out[1] = %f, want 1.03", out[1])
	}
}

func TestFrame(t *testing.T) {
	samples := make([]float64, 100)
	for i := range samples {
		samples[i] = float64(i)
	}
	frames := Frame(samples, 25, 10)
	// numFrames = 1 + (100-25)/10 = 8
	if len(frames) != 8 {
		t.Fatalf("numFrames = %d, want 8", len(frames))
	}
	if len(frames[0]) != 25 {
		t.Fatalf("frameLen = %d, want 25", len(frames[0]))
	}
	// Second frame starts at index 10
	if frames[1][0] != 10.0 {
		t.Errorf("frames[1][0] = %f, want 10.0", frames[1][0])
	}
}

func TestHammingWindow(t *testing.T) {
	frame := make([]float64, 10)
	for i := range frame {
		frame[i] = 1.0
	}
	HammingWindow(frame)
	// Hamming at endpoints should be ~0.08
	if math.Abs(frame[0]-0.08) > 0.01 {
		t.Errorf("frame[0] = %f, want ~0.08", frame[0])
	}
	// Hamming at midpoint should be ~1.0
	mid := len(frame) / 2
	if frame[mid] < 0.9 {
		t.Errorf("frame[%d] = %f, want close to 1.0", mid, frame[mid])
	}
}

func TestFFT_KnownInput(t *testing.T) {
	// FFT of [1, 0, 0, 0, 0, 0, 0, 0] = [1, 1, 1, 1, 1, 1, 1, 1]
	x := make([]complex128, 8)
	x[0] = 1
	X := FFT(x)
	for i, v := range X {
		if cmplx.Abs(v-1) > 1e-10 {
			t.Errorf("X[%d] = %v, want 1+0i", i, v)
		}
	}
}

func TestFFT_Sinusoid(t *testing.T) {
	// 8-point FFT of a pure cosine at bin 2
	n := 8
	x := make([]complex128, n)
	for i := 0; i < n; i++ {
		x[i] = complex(math.Cos(2*math.Pi*2*float64(i)/float64(n)), 0)
	}
	X := FFT(x)
	// Should have peaks at bin 2 and bin 6 (N-2)
	for i := 0; i < n; i++ {
		mag := cmplx.Abs(X[i])
		if i == 2 || i == 6 {
			if mag < 3.0 { // should be N/2 = 4
				t.Errorf("|X[%d]| = %f, want ~4.0", i, mag)
			}
		} else {
			if mag > 1e-10 {
				t.Errorf("|X[%d]| = %f, want ~0.0", i, mag)
			}
		}
	}
}

func TestPowerSpectrum(t *testing.T) {
	frame := make([]float64, 16)
	frame[0] = 1.0 // impulse
	ps := PowerSpectrum(frame, 16)
	// Power spectrum of impulse should be flat: 1/N = 0.0625
	if len(ps) != 9 { // 16/2+1
		t.Fatalf("len(ps) = %d, want 9", len(ps))
	}
	for i, v := range ps {
		if math.Abs(v-1.0/16.0) > 1e-10 {
			t.Errorf("ps[%d] = %f, want %f", i, v, 1.0/16.0)
		}
	}
}

func TestMelFilterbank(t *testing.T) {
	fb := NewMelFilterbank(26, 512, 16000, 0, 8000, 1.0)
	if len(fb.Filters) != 26 {
		t.Fatalf("numFilters = %d, want 26", len(fb.Filters))
	}
	// Each filter should be length 257 (512/2+1)
	for i, f := range fb.Filters {
		if len(f) != 257 {
			t.Fatalf("filter[%d] len = %d, want 257", i, len(f))
		}
	}
	// Filters should be non-negative
	for i, f := range fb.Filters {
		for j, v := range f {
			if v < 0 {
				t.Errorf("filter[%d][%d] = %f < 0", i, j, v)
			}
		}
	}
}

func TestDCT(t *testing.T) {
	// DCT of constant input should have energy only in the 0th coefficient
	input := make([]float64, 26)
	for i := range input {
		input[i] = 1.0
	}
	cepstra := DCT(input, 13)
	if len(cepstra) != 13 {
		t.Fatalf("len(cepstra) = %d, want 13", len(cepstra))
	}
	// c[0] should be sum of input = 26
	if math.Abs(cepstra[0]-26.0) > 1e-10 {
		t.Errorf("cepstra[0] = %f, want 26.0", cepstra[0])
	}
	// Other coefficients should be near zero
	for k := 1; k < 13; k++ {
		if math.Abs(cepstra[k]) > 1e-10 {
			t.Errorf("cepstra[%d] = %f, want ~0", k, cepstra[k])
		}
	}
}

func TestDelta(t *testing.T) {
	// Linear ramp: features[t] = [t]
	features := make([][]float64, 10)
	for t := range features {
		features[t] = []float64{float64(t)}
	}
	d := Delta(features, 2)
	if len(d) != 10 {
		t.Fatalf("len(d) = %d, want 10", len(d))
	}
	// Delta of a linear ramp should be constant ~1.0 (in the middle frames)
	for i := 2; i < 8; i++ {
		if math.Abs(d[i][0]-1.0) > 1e-10 {
			t.Errorf("delta[%d] = %f, want 1.0", i, d[i][0])
		}
	}
}

func TestExtract_Dimensions(t *testing.T) {
	cfg := DefaultConfig()
	// Generate 1 second of 440Hz sine at 16kHz
	n := 16000
	samples := make([]float64, n)
	for i := range samples {
		samples[i] = math.Sin(2 * math.Pi * 440 * float64(i) / 16000)
	}
	mfccs, err := Extract(samples, cfg)
	if err != nil {
		t.Fatalf("Extract error: %v", err)
	}
	// Expected frames: 1 + (16000 - 400) / 160 = 98 (approx)
	frameLen := int(cfg.FrameLenMs * float64(cfg.SampleRate) / 1000.0)
	frameShift := int(cfg.FrameShiftMs * float64(cfg.SampleRate) / 1000.0)
	expectedFrames := 1 + (n-frameLen)/frameShift
	if len(mfccs) != expectedFrames {
		t.Errorf("numFrames = %d, want %d", len(mfccs), expectedFrames)
	}
	// Feature dim with delta+deltadelta: 13*3 = 39
	expectedDim := cfg.FeatureDim()
	if expectedDim != 39 {
		t.Errorf("FeatureDim = %d, want 39", expectedDim)
	}
	if len(mfccs[0]) != expectedDim {
		t.Errorf("feature dim = %d, want %d", len(mfccs[0]), expectedDim)
	}
	// Values should be finite
	for i, frame := range mfccs {
		for j, v := range frame {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Errorf("mfcc[%d][%d] = %f (not finite)", i, j, v)
			}
		}
	}
}

func TestExtract_EmptySamples(t *testing.T) {
	cfg := DefaultConfig()
	_, err := Extract(nil, cfg)
	if err == nil {
		t.Fatal("expected error for empty samples")
	}
}

func TestWarpFreq_Identity(t *testing.T) {
	// alpha=1.0 should return the input frequency unchanged
	for _, f := range []float64{0, 50, 100, 1000, 4000, 7500, 8000} {
		got := WarpFreq(f, 1.0, 100, 7500, 0, 8000)
		if math.Abs(got-f) > 1e-10 {
			t.Errorf("WarpFreq(%f, 1.0) = %f, want %f", f, got, f)
		}
	}
}

func TestWarpFreq_Boundaries(t *testing.T) {
	// W(lowFreq) = lowFreq and W(highFreq) = highFreq for any alpha
	for _, alpha := range []float64{0.82, 0.9, 1.0, 1.1, 1.2} {
		low := WarpFreq(0, alpha, 100, 7500, 0, 8000)
		if math.Abs(low-0) > 1e-10 {
			t.Errorf("WarpFreq(0, %f) = %f, want 0", alpha, low)
		}
		high := WarpFreq(8000, alpha, 100, 7500, 0, 8000)
		if math.Abs(high-8000) > 1e-10 {
			t.Errorf("WarpFreq(8000, %f) = %f, want 8000", alpha, high)
		}
	}
}

func TestWarpFreq_MiddleRegion(t *testing.T) {
	// In [vtlnLow, vtlnHigh], W(f) = f * alpha
	alpha := 0.9
	for _, f := range []float64{100, 500, 2000, 5000, 7499} {
		got := WarpFreq(f, alpha, 100, 7500, 0, 8000)
		want := f * alpha
		if math.Abs(got-want) > 1e-10 {
			t.Errorf("WarpFreq(%f, %f) = %f, want %f", f, alpha, got, want)
		}
	}
}

func TestMelFilterbank_VTLN(t *testing.T) {
	// Warped filterbank should differ from unwarped
	fb1 := NewMelFilterbank(26, 512, 16000, 0, 8000, 1.0)
	fb09 := NewMelFilterbank(26, 512, 16000, 0, 8000, 0.9)

	differs := false
	for i := range fb1.Filters {
		for j := range fb1.Filters[i] {
			if fb1.Filters[i][j] != fb09.Filters[i][j] {
				differs = true
				break
			}
		}
		if differs {
			break
		}
	}
	if !differs {
		t.Error("warped filterbank (alpha=0.9) is identical to unwarped")
	}
}

func TestExtractWithVTLN(t *testing.T) {
	cfg := DefaultConfig()
	n := 16000
	samples := make([]float64, n)
	for i := range samples {
		samples[i] = math.Sin(2 * math.Pi * 440 * float64(i) / 16000)
	}

	// Use a simple scorer that returns sum of first cepstral coefficients
	scorer := func(feats [][]float64) float64 {
		sum := 0.0
		for _, f := range feats {
			sum += f[0]
		}
		return sum
	}

	feats, alpha, err := ExtractWithVTLN(samples, cfg, scorer)
	if err != nil {
		t.Fatalf("ExtractWithVTLN error: %v", err)
	}
	if alpha < 0.82 || alpha > 1.20 {
		t.Errorf("alpha = %f, out of range [0.82, 1.20]", alpha)
	}
	expectedDim := cfg.FeatureDim()
	if len(feats[0]) != expectedDim {
		t.Errorf("feature dim = %d, want %d", len(feats[0]), expectedDim)
	}
	// Values should be finite
	for i, frame := range feats {
		for j, v := range frame {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Errorf("feat[%d][%d] = %f (not finite)", i, j, v)
			}
		}
	}
}

func TestFeaturesFromSpectra_MatchesExtract(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Alpha = 1.0
	n := 16000
	samples := make([]float64, n)
	for i := range samples {
		samples[i] = math.Sin(2 * math.Pi * 440 * float64(i) / 16000)
	}

	direct, err := Extract(samples, cfg)
	if err != nil {
		t.Fatalf("Extract error: %v", err)
	}

	spectra, err := ExtractPowerSpectra(samples, cfg)
	if err != nil {
		t.Fatalf("ExtractPowerSpectra error: %v", err)
	}
	fromSpectra := FeaturesFromSpectra(spectra, cfg)

	if len(direct) != len(fromSpectra) {
		t.Fatalf("frame count mismatch: %d vs %d", len(direct), len(fromSpectra))
	}
	for i := range direct {
		if len(direct[i]) != len(fromSpectra[i]) {
			t.Fatalf("dim mismatch at frame %d: %d vs %d", i, len(direct[i]), len(fromSpectra[i]))
		}
		for j := range direct[i] {
			if math.Abs(direct[i][j]-fromSpectra[i][j]) > 1e-10 {
				t.Errorf("mismatch at [%d][%d]: %f vs %f", i, j, direct[i][j], fromSpectra[i][j])
			}
		}
	}
}
