//go:build amd64

package blas

// dotAVX2 computes the dot product of two float64 vectors using AVX2+FMA.
// a and b must point to at least n contiguous float64 values.
func dotAVX2(a, b *float64, n int) float64
