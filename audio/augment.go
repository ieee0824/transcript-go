package audio

// SpeedPerturb resamples the input audio samples by the given speed factor.
// A factor > 1.0 makes the audio faster (shorter, higher pitch).
// A factor < 1.0 makes the audio slower (longer, lower pitch).
// The sample rate is unchanged; the returned slice has length int(len(samples) / factor).
// Linear interpolation is used between samples.
func SpeedPerturb(samples []float64, factor float64) []float64 {
	if len(samples) == 0 || factor <= 0 {
		return nil
	}

	origLen := len(samples)
	newLen := int(float64(origLen) / factor)
	if newLen == 0 {
		return nil
	}

	result := make([]float64, newLen)
	for i := 0; i < newLen; i++ {
		srcIdx := float64(i) * factor
		idx0 := int(srcIdx)
		frac := srcIdx - float64(idx0)

		if idx0+1 < origLen {
			result[i] = samples[idx0]*(1.0-frac) + samples[idx0+1]*frac
		} else if idx0 < origLen {
			result[i] = samples[idx0]
		}
	}

	return result
}
