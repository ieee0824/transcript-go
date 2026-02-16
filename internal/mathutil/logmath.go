package mathutil

import "math"

// LogZero represents log(0), used as negative infinity in log-domain arithmetic.
const LogZero = -1e30

// LogAdd returns log(exp(a) + exp(b)) in a numerically stable way.
// Uses threshold-based early exit to skip expensive exp/log1p when the
// smaller value contributes less than float64 precision (exp(-36) ≈ 2.3e-16).
func LogAdd(a, b float64) float64 {
	if a > b {
		if b == LogZero {
			return a
		}
		d := b - a
		if d < -36.0 {
			return a
		}
		return a + math.Log1p(math.Exp(d))
	}
	if a == LogZero {
		return b
	}
	d := a - b
	if d < -36.0 {
		return b
	}
	return b + math.Log1p(math.Exp(d))
}

// LogSub returns log(exp(a) - exp(b)), assuming a > b.
func LogSub(a, b float64) float64 {
	if b == LogZero {
		return a
	}
	if a <= b {
		return LogZero
	}
	return a + math.Log1p(-math.Exp(b-a))
}
