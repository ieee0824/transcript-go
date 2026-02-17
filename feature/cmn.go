package feature

// ApplyCMN subtracts the utterance-level mean from each feature dimension (Cepstral Mean Normalization).
// This removes channel and speaker-dependent spectral bias.
func ApplyCMN(features [][]float64) {
	T := len(features)
	if T == 0 {
		return
	}
	dim := len(features[0])
	mean := make([]float64, dim)
	for t := 0; t < T; t++ {
		for d := 0; d < dim; d++ {
			mean[d] += features[t][d]
		}
	}
	invT := 1.0 / float64(T)
	for d := 0; d < dim; d++ {
		mean[d] *= invT
	}
	for t := 0; t < T; t++ {
		for d := 0; d < dim; d++ {
			features[t][d] -= mean[d]
		}
	}
}
