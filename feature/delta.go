package feature

// Delta computes delta (first derivative) coefficients with window N.
// Uses the regression formula: d[t] = sum_{n=1}^{N} n*(c[t+n] - c[t-n]) / (2 * sum_{n=1}^{N} n^2)
func Delta(features [][]float64, N int) [][]float64 {
	T := len(features)
	if T == 0 {
		return nil
	}
	dim := len(features[0])
	deltas := make([][]float64, T)

	// Precompute denominator
	denom := 0.0
	for n := 1; n <= N; n++ {
		denom += float64(n * n)
	}
	denom *= 2.0

	buf := make([]float64, T*dim)
	for t := 0; t < T; t++ {
		deltas[t] = buf[t*dim : (t+1)*dim]
		for d := 0; d < dim; d++ {
			num := 0.0
			for n := 1; n <= N; n++ {
				// Clamp indices to valid range
				tp := t + n
				if tp >= T {
					tp = T - 1
				}
				tn := t - n
				if tn < 0 {
					tn = 0
				}
				num += float64(n) * (features[tp][d] - features[tn][d])
			}
			deltas[t][d] = num / denom
		}
	}
	return deltas
}

// AppendDeltas appends delta and delta-delta columns to each frame.
// Input: [T][D] -> Output: [T][3*D]
func AppendDeltas(features [][]float64) [][]float64 {
	d1 := Delta(features, 2)
	d2 := Delta(d1, 2)

	T := len(features)
	dim := len(features[0])
	out := make([][]float64, T)
	rowBuf := make([]float64, T*dim*3)
	for t := 0; t < T; t++ {
		row := rowBuf[t*dim*3 : (t+1)*dim*3]
		copy(row[:dim], features[t])
		copy(row[dim:dim*2], d1[t])
		copy(row[dim*2:], d2[t])
		out[t] = row
	}
	return out
}
