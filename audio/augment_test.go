package audio

import (
	"math"
	"testing"
)

func TestSpeedPerturb_Identity(t *testing.T) {
	samples := []float64{0.0, 0.25, 0.5, 0.75, 1.0}
	result := SpeedPerturb(samples, 1.0)
	if len(result) != len(samples) {
		t.Fatalf("len = %d, want %d", len(result), len(samples))
	}
	for i, v := range result {
		if math.Abs(v-samples[i]) > 1e-10 {
			t.Errorf("result[%d] = %f, want %f", i, v, samples[i])
		}
	}
}

func TestSpeedPerturb_SlowerProducesLonger(t *testing.T) {
	n := 16000
	samples := make([]float64, n)
	for i := range samples {
		samples[i] = math.Sin(2 * math.Pi * 440 * float64(i) / 16000)
	}
	result := SpeedPerturb(samples, 0.9)
	expectedLen := int(float64(n) / 0.9)
	if len(result) != expectedLen {
		t.Errorf("len = %d, want %d", len(result), expectedLen)
	}
	for i, v := range result {
		if v < -1.01 || v > 1.01 {
			t.Errorf("result[%d] = %f, out of range [-1,1]", i, v)
			break
		}
	}
}

func TestSpeedPerturb_FasterProducesShorter(t *testing.T) {
	n := 16000
	samples := make([]float64, n)
	for i := range samples {
		samples[i] = math.Sin(2 * math.Pi * 440 * float64(i) / 16000)
	}
	result := SpeedPerturb(samples, 1.1)
	expectedLen := int(float64(n) / 1.1)
	if len(result) != expectedLen {
		t.Errorf("len = %d, want %d", len(result), expectedLen)
	}
}

func TestSpeedPerturb_EmptyInput(t *testing.T) {
	result := SpeedPerturb(nil, 1.0)
	if result != nil {
		t.Errorf("expected nil for empty input, got len=%d", len(result))
	}
}

func TestSpeedPerturb_InvalidFactor(t *testing.T) {
	samples := []float64{1.0, 2.0, 3.0}
	if result := SpeedPerturb(samples, 0.0); result != nil {
		t.Error("expected nil for factor=0")
	}
	if result := SpeedPerturb(samples, -1.0); result != nil {
		t.Error("expected nil for factor<0")
	}
}

func TestSpeedPerturb_LinearInterpolation(t *testing.T) {
	// Ramp [0, 1, 2, 3, 4] with factor 0.5 (2x longer)
	// output[i] maps to source position 0.5*i
	samples := []float64{0.0, 1.0, 2.0, 3.0, 4.0}
	result := SpeedPerturb(samples, 0.5)
	expectedLen := int(5.0 / 0.5) // 10
	if len(result) != expectedLen {
		t.Fatalf("len = %d, want %d", len(result), expectedLen)
	}
	for i := 0; i < 9; i++ {
		want := float64(i) * 0.5
		if math.Abs(result[i]-want) > 1e-10 {
			t.Errorf("result[%d] = %f, want %f", i, result[i], want)
		}
	}
}
