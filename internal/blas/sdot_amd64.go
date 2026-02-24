//go:build amd64

package blas

// dotAVX2f32 computes the dot product of two float32 vectors using AVX2+FMA.
// a and b must point to at least n contiguous float32 values.
func dotAVX2f32(a, b *float32, n int) float32
