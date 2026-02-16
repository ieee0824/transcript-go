package mathutil

import (
	"math"
	"testing"
)

func TestNewMat(t *testing.T) {
	m := NewMat(3, 4)
	if len(m) != 3 {
		t.Fatalf("rows = %d, want 3", len(m))
	}
	for i, row := range m {
		if len(row) != 4 {
			t.Fatalf("row %d cols = %d, want 4", i, len(row))
		}
	}
}

func TestNewMatFill(t *testing.T) {
	m := NewMatFill(2, 3, 1.5)
	for i, row := range m {
		for j, v := range row {
			if v != 1.5 {
				t.Errorf("m[%d][%d] = %f, want 1.5", i, j, v)
			}
		}
	}
}

func TestDotVec(t *testing.T) {
	a := Vec{1, 2, 3}
	b := Vec{4, 5, 6}
	got := DotVec(a, b)
	want := 32.0
	if math.Abs(got-want) > 1e-10 {
		t.Errorf("DotVec = %f, want %f", got, want)
	}
}

func TestAddVec(t *testing.T) {
	a := Vec{1, 2, 3}
	b := Vec{4, 5, 6}
	dst := NewVec(3)
	AddVec(dst, a, b)
	want := Vec{5, 7, 9}
	for i := range dst {
		if dst[i] != want[i] {
			t.Errorf("dst[%d] = %f, want %f", i, dst[i], want[i])
		}
	}
}

func TestScaleVec(t *testing.T) {
	src := Vec{1, 2, 3}
	dst := NewVec(3)
	ScaleVec(dst, 2.0, src)
	want := Vec{2, 4, 6}
	for i := range dst {
		if dst[i] != want[i] {
			t.Errorf("dst[%d] = %f, want %f", i, dst[i], want[i])
		}
	}
}
