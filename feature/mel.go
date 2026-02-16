package feature

import "math"

// sparseFilter stores only the non-zero range of a triangular filter.
type sparseFilter struct {
	start  int       // first non-zero bin index
	coeffs []float64 // non-zero coefficient values
}

// MelFilterbank represents the triangular Mel-spaced filterbank.
type MelFilterbank struct {
	Filters [][]float64    // [numFilters][fftSize/2+1]
	sparse  []sparseFilter // sparse representation for fast inner loop
}

// NewMelFilterbank constructs the filterbank.
func NewMelFilterbank(numFilters, fftSize, sampleRate int, lowFreq, highFreq float64) *MelFilterbank {
	nBins := fftSize/2 + 1
	lowMel := hzToMel(lowFreq)
	highMel := hzToMel(highFreq)

	// Create numFilters+2 equally spaced points on the Mel scale
	melPoints := make([]float64, numFilters+2)
	step := (highMel - lowMel) / float64(numFilters+1)
	for i := range melPoints {
		melPoints[i] = lowMel + float64(i)*step
	}

	// Convert to Hz and then to FFT bin indices
	binIndices := make([]int, numFilters+2)
	for i, m := range melPoints {
		freq := melToHz(m)
		binIndices[i] = int(math.Floor(freq * float64(fftSize+1) / float64(sampleRate)))
	}

	// Build triangular filters
	filters := make([][]float64, numFilters)
	for i := 0; i < numFilters; i++ {
		filters[i] = make([]float64, nBins)
		left := binIndices[i]
		center := binIndices[i+1]
		right := binIndices[i+2]

		for j := left; j < center && j < nBins; j++ {
			if center != left {
				filters[i][j] = float64(j-left) / float64(center-left)
			}
		}
		for j := center; j <= right && j < nBins; j++ {
			if right != center {
				filters[i][j] = float64(right-j) / float64(right-center)
			}
		}
	}

	fb := &MelFilterbank{Filters: filters}

	// Build sparse representation: only store non-zero coefficients per filter
	fb.sparse = make([]sparseFilter, numFilters)
	for i, f := range filters {
		start, end := 0, 0
		found := false
		for j, v := range f {
			if v != 0 {
				if !found {
					start = j
					found = true
				}
				end = j + 1
			}
		}
		if found {
			fb.sparse[i] = sparseFilter{
				start:  start,
				coeffs: make([]float64, end-start),
			}
			copy(fb.sparse[i].coeffs, f[start:end])
		}
	}

	return fb
}

// Apply multiplies the power spectrum through each filter and returns log Mel energies.
func (fb *MelFilterbank) Apply(powerSpec []float64) []float64 {
	energies := make([]float64, len(fb.sparse))
	fb.applyInto(powerSpec, energies)
	return energies
}

// applyInto writes log Mel energies into dst using sparse representation (no allocation).
func (fb *MelFilterbank) applyInto(powerSpec, dst []float64) {
	for i, sf := range fb.sparse {
		sum := 0.0
		end := sf.start + len(sf.coeffs)
		if end > len(powerSpec) {
			end = len(powerSpec)
		}
		ps := powerSpec[sf.start:end]
		coeffs := sf.coeffs[:len(ps)]
		for j, p := range ps {
			sum += p * coeffs[j]
		}
		if sum < 1e-30 {
			sum = 1e-30
		}
		dst[i] = math.Log(sum)
	}
}

// DCT applies Type-II DCT to extract cepstral coefficients.
func DCT(logMelEnergies []float64, numCepstra int) []float64 {
	n := len(logMelEnergies)
	cepstra := make([]float64, numCepstra)
	for k := 0; k < numCepstra; k++ {
		sum := 0.0
		for j := 0; j < n; j++ {
			sum += logMelEnergies[j] * math.Cos(math.Pi*float64(k)*(float64(j)+0.5)/float64(n))
		}
		cepstra[k] = sum
	}
	return cepstra
}

// dctTable holds precomputed cosine values for DCT.
type dctTable struct {
	cos [][]float64 // [numCepstra][numFilters]
}

func newDCTTable(numCepstra, numFilters int) *dctTable {
	t := &dctTable{cos: make([][]float64, numCepstra)}
	for k := 0; k < numCepstra; k++ {
		t.cos[k] = make([]float64, numFilters)
		for j := 0; j < numFilters; j++ {
			t.cos[k][j] = math.Cos(math.Pi * float64(k) * (float64(j) + 0.5) / float64(numFilters))
		}
	}
	return t
}

// applyInto computes DCT into dst using precomputed cosine table (no allocation).
func (t *dctTable) applyInto(logMelEnergies, dst []float64) {
	for k := range t.cos {
		sum := 0.0
		row := t.cos[k]
		for j, c := range row {
			sum += logMelEnergies[j] * c
		}
		dst[k] = sum
	}
}

// lifterTable holds precomputed sinusoidal liftering coefficients.
type lifterTable struct {
	coeff []float64
}

func newLifterTable(numCepstra, L int) *lifterTable {
	t := &lifterTable{coeff: make([]float64, numCepstra)}
	for i := range t.coeff {
		t.coeff[i] = 1.0 + float64(L)/2.0*math.Sin(math.Pi*float64(i)/float64(L))
	}
	return t
}

func (t *lifterTable) apply(cepstra []float64) {
	for i := range cepstra {
		cepstra[i] *= t.coeff[i]
	}
}

// CepstralLifter applies sinusoidal liftering to cepstral coefficients.
func CepstralLifter(cepstra []float64, L int) {
	for i := range cepstra {
		lift := 1.0 + float64(L)/2.0*math.Sin(math.Pi*float64(i)/float64(L))
		cepstra[i] *= lift
	}
}

func hzToMel(hz float64) float64 {
	return 2595.0 * math.Log10(1.0+hz/700.0)
}

func melToHz(mel float64) float64 {
	return 700.0 * (math.Pow(10, mel/2595.0) - 1.0)
}
