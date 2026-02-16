package mathutil

import (
	"math"
	"testing"
)

func TestLogAdd(t *testing.T) {
	// log(exp(log(2)) + exp(log(3))) = log(5)
	a := math.Log(2)
	b := math.Log(3)
	got := LogAdd(a, b)
	want := math.Log(5)
	if math.Abs(got-want) > 1e-10 {
		t.Errorf("LogAdd(log(2), log(3)) = %f, want %f", got, want)
	}
}

func TestLogAddWithLogZero(t *testing.T) {
	a := math.Log(5)
	if got := LogAdd(LogZero, a); math.Abs(got-a) > 1e-10 {
		t.Errorf("LogAdd(LogZero, %f) = %f, want %f", a, got, a)
	}
	if got := LogAdd(a, LogZero); math.Abs(got-a) > 1e-10 {
		t.Errorf("LogAdd(%f, LogZero) = %f, want %f", a, got, a)
	}
}

func TestLogSub(t *testing.T) {
	// log(exp(log(5)) - exp(log(3))) = log(2)
	a := math.Log(5)
	b := math.Log(3)
	got := LogSub(a, b)
	want := math.Log(2)
	if math.Abs(got-want) > 1e-10 {
		t.Errorf("LogSub(log(5), log(3)) = %f, want %f", got, want)
	}
}

func TestLogSubWithLogZero(t *testing.T) {
	a := math.Log(5)
	if got := LogSub(a, LogZero); math.Abs(got-a) > 1e-10 {
		t.Errorf("LogSub(%f, LogZero) = %f, want %f", a, got, a)
	}
}
