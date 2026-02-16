package mathutil

// Vec is a float64 vector.
type Vec = []float64

// Mat is a 2D float64 matrix stored as row-major [][]float64.
type Mat = [][]float64

// NewMat creates a rows x cols matrix initialized to zero.
func NewMat(rows, cols int) Mat {
	m := make(Mat, rows)
	data := make([]float64, rows*cols)
	for i := range m {
		m[i] = data[i*cols : (i+1)*cols]
	}
	return m
}

// NewMatFill creates a rows x cols matrix filled with val.
func NewMatFill(rows, cols int, val float64) Mat {
	m := NewMat(rows, cols)
	for i := range m {
		for j := range m[i] {
			m[i][j] = val
		}
	}
	return m
}

// DotVec returns the dot product of a and b.
func DotVec(a, b Vec) float64 {
	sum := 0.0
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// AddVec stores a+b in dst.
func AddVec(dst, a, b Vec) {
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
}

// ScaleVec stores alpha*src in dst.
func ScaleVec(dst Vec, alpha float64, src Vec) {
	for i := range dst {
		dst[i] = alpha * src[i]
	}
}

// CopyVec copies src into dst.
func CopyVec(dst, src Vec) {
	copy(dst, src)
}

// NewVec creates a vector of length n initialized to zero.
func NewVec(n int) Vec {
	return make(Vec, n)
}

// NewVecFill creates a vector of length n filled with val.
func NewVecFill(n int, val float64) Vec {
	v := make(Vec, n)
	for i := range v {
		v[i] = val
	}
	return v
}

// FillMat fills all elements of an existing matrix with val.
func FillMat(m Mat, val float64) {
	for i := range m {
		for j := range m[i] {
			m[i][j] = val
		}
	}
}

// FillVec fills all elements of an existing vector with val.
func FillVec(v Vec, val float64) {
	for i := range v {
		v[i] = val
	}
}
