package audio

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"
)

// buildWAV constructs a minimal valid WAV file in memory.
func buildWAV(sampleRate uint32, bitsPerSample, numChannels uint16, samples []int16) []byte {
	var buf bytes.Buffer
	dataSize := uint32(len(samples) * 2)
	byteRate := sampleRate * uint32(numChannels) * uint32(bitsPerSample) / 8
	blockAlign := numChannels * bitsPerSample / 8

	// RIFF header
	buf.WriteString("RIFF")
	binary.Write(&buf, binary.LittleEndian, uint32(36+dataSize))
	buf.WriteString("WAVE")

	// fmt chunk
	buf.WriteString("fmt ")
	binary.Write(&buf, binary.LittleEndian, uint32(16))    // chunk size
	binary.Write(&buf, binary.LittleEndian, uint16(1))     // PCM
	binary.Write(&buf, binary.LittleEndian, numChannels)
	binary.Write(&buf, binary.LittleEndian, sampleRate)
	binary.Write(&buf, binary.LittleEndian, byteRate)
	binary.Write(&buf, binary.LittleEndian, blockAlign)
	binary.Write(&buf, binary.LittleEndian, bitsPerSample)

	// data chunk
	buf.WriteString("data")
	binary.Write(&buf, binary.LittleEndian, dataSize)
	binary.Write(&buf, binary.LittleEndian, samples)

	return buf.Bytes()
}

func TestReadWAV_Valid(t *testing.T) {
	// Generate a 440Hz sine wave, 100 samples at 16kHz
	n := 100
	raw := make([]int16, n)
	for i := range raw {
		raw[i] = int16(16000 * math.Sin(2*math.Pi*440*float64(i)/16000))
	}

	data := buildWAV(16000, 16, 1, raw)
	r := bytes.NewReader(data)

	samples, header, err := ReadWAV(r)
	if err != nil {
		t.Fatalf("ReadWAV error: %v", err)
	}

	if header.SampleRate != 16000 {
		t.Errorf("SampleRate = %d, want 16000", header.SampleRate)
	}
	if header.NumChannels != 1 {
		t.Errorf("NumChannels = %d, want 1", header.NumChannels)
	}
	if header.BitsPerSample != 16 {
		t.Errorf("BitsPerSample = %d, want 16", header.BitsPerSample)
	}
	if header.NumSamples != n {
		t.Errorf("NumSamples = %d, want %d", header.NumSamples, n)
	}
	if len(samples) != n {
		t.Fatalf("len(samples) = %d, want %d", len(samples), n)
	}

	// Verify conversion: int16 -> float64
	for i := 0; i < n; i++ {
		want := float64(raw[i]) / 32768.0
		if math.Abs(samples[i]-want) > 1e-10 {
			t.Errorf("samples[%d] = %f, want %f", i, samples[i], want)
		}
	}
}

func TestReadWAV_NotRIFF(t *testing.T) {
	data := []byte("NOT_RIFF_DATA_HERE_EXTRA")
	r := bytes.NewReader(data)
	_, _, err := ReadWAV(r)
	if err == nil {
		t.Fatal("expected error for non-RIFF data")
	}
}

func TestReadWAV_UnsupportedSampleRate(t *testing.T) {
	raw := []int16{0, 0, 0, 0}
	data := buildWAV(44100, 16, 1, raw)
	r := bytes.NewReader(data)
	_, _, err := ReadWAV(r)
	if err == nil {
		t.Fatal("expected error for 44100 sample rate")
	}
}

func TestReadWAV_UnsupportedStereo(t *testing.T) {
	raw := []int16{0, 0, 0, 0}
	data := buildWAV(16000, 16, 2, raw)
	r := bytes.NewReader(data)
	_, _, err := ReadWAV(r)
	if err == nil {
		t.Fatal("expected error for stereo")
	}
}
