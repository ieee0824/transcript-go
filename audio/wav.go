package audio

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
)

// WAVHeader holds the parsed RIFF/WAV header fields.
type WAVHeader struct {
	SampleRate    uint32
	BitsPerSample uint16
	NumChannels   uint16
	NumSamples    int
}

// ReadWAV reads a WAV file and returns normalized float64 samples in [-1.0, 1.0].
// It returns an error if the format is not 16-bit PCM mono at 16kHz.
func ReadWAV(r io.ReadSeeker) ([]float64, WAVHeader, error) {
	var header WAVHeader

	// Read RIFF header
	var riffID [4]byte
	if err := binary.Read(r, binary.LittleEndian, &riffID); err != nil {
		return nil, header, fmt.Errorf("read RIFF ID: %w", err)
	}
	if string(riffID[:]) != "RIFF" {
		return nil, header, errors.New("not a RIFF file")
	}

	var fileSize uint32
	if err := binary.Read(r, binary.LittleEndian, &fileSize); err != nil {
		return nil, header, fmt.Errorf("read file size: %w", err)
	}

	var waveID [4]byte
	if err := binary.Read(r, binary.LittleEndian, &waveID); err != nil {
		return nil, header, fmt.Errorf("read WAVE ID: %w", err)
	}
	if string(waveID[:]) != "WAVE" {
		return nil, header, errors.New("not a WAVE file")
	}

	// Read chunks
	var fmtFound, dataFound bool
	var samples []float64

	for {
		var chunkID [4]byte
		if err := binary.Read(r, binary.LittleEndian, &chunkID); err != nil {
			if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
				break
			}
			return nil, header, fmt.Errorf("read chunk ID: %w", err)
		}

		var chunkSize uint32
		if err := binary.Read(r, binary.LittleEndian, &chunkSize); err != nil {
			return nil, header, fmt.Errorf("read chunk size: %w", err)
		}

		switch string(chunkID[:]) {
		case "fmt ":
			if err := readFmtChunk(r, chunkSize, &header); err != nil {
				return nil, header, err
			}
			fmtFound = true

		case "data":
			if !fmtFound {
				return nil, header, errors.New("data chunk before fmt chunk")
			}
			var err error
			samples, err = readDataChunk(r, chunkSize, &header)
			if err != nil {
				return nil, header, err
			}
			dataFound = true

		default:
			// Skip unknown chunks; align to even boundary
			skip := int64(chunkSize)
			if chunkSize%2 != 0 {
				skip++
			}
			if _, err := r.Seek(skip, io.SeekCurrent); err != nil {
				return nil, header, fmt.Errorf("skip chunk %q: %w", chunkID, err)
			}
		}

		if fmtFound && dataFound {
			break
		}
	}

	if !fmtFound {
		return nil, header, errors.New("missing fmt chunk")
	}
	if !dataFound {
		return nil, header, errors.New("missing data chunk")
	}

	return samples, header, nil
}

// ReadWAVFile is a convenience wrapper that opens a file path.
func ReadWAVFile(path string) ([]float64, WAVHeader, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, WAVHeader{}, err
	}
	defer f.Close()
	return ReadWAV(f)
}

func readFmtChunk(r io.ReadSeeker, size uint32, h *WAVHeader) error {
	var audioFormat uint16
	if err := binary.Read(r, binary.LittleEndian, &audioFormat); err != nil {
		return fmt.Errorf("read audio format: %w", err)
	}
	if audioFormat != 1 {
		return fmt.Errorf("unsupported audio format %d (only PCM=1 supported)", audioFormat)
	}

	if err := binary.Read(r, binary.LittleEndian, &h.NumChannels); err != nil {
		return fmt.Errorf("read num channels: %w", err)
	}
	if h.NumChannels != 1 {
		return fmt.Errorf("unsupported channel count %d (only mono supported)", h.NumChannels)
	}

	if err := binary.Read(r, binary.LittleEndian, &h.SampleRate); err != nil {
		return fmt.Errorf("read sample rate: %w", err)
	}
	if h.SampleRate != 16000 {
		return fmt.Errorf("unsupported sample rate %d (only 16000 supported)", h.SampleRate)
	}

	// Skip byteRate (4 bytes) and blockAlign (2 bytes)
	if _, err := r.Seek(6, io.SeekCurrent); err != nil {
		return fmt.Errorf("skip byte rate / block align: %w", err)
	}

	if err := binary.Read(r, binary.LittleEndian, &h.BitsPerSample); err != nil {
		return fmt.Errorf("read bits per sample: %w", err)
	}
	if h.BitsPerSample != 16 {
		return fmt.Errorf("unsupported bits per sample %d (only 16 supported)", h.BitsPerSample)
	}

	// Skip any extra fmt bytes
	consumed := uint32(16) // audioFormat(2) + numChannels(2) + sampleRate(4) + byteRate(4) + blockAlign(2) + bitsPerSample(2)
	if size > consumed {
		if _, err := r.Seek(int64(size-consumed), io.SeekCurrent); err != nil {
			return fmt.Errorf("skip extra fmt bytes: %w", err)
		}
	}

	return nil
}

func readDataChunk(r io.Reader, size uint32, h *WAVHeader) ([]float64, error) {
	bytesPerSample := int(h.BitsPerSample) / 8
	numSamples := int(size) / bytesPerSample
	h.NumSamples = numSamples

	raw := make([]int16, numSamples)
	if err := binary.Read(r, binary.LittleEndian, raw); err != nil {
		return nil, fmt.Errorf("read PCM data: %w", err)
	}

	samples := make([]float64, numSamples)
	for i, s := range raw {
		samples[i] = float64(s) / 32768.0
	}

	return samples, nil
}
