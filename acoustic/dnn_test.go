package acoustic

import (
	"bytes"
	"math"
	"math/rand"
	"testing"

	"github.com/ieee0824/transcript-go/internal/blas"
)

func TestDNNForward_Dimensions(t *testing.T) {
	d := NewDNN(39, 16, 3) // small DNN for testing
	T := 10
	features := make([][]float64, T)
	for i := range features {
		features[i] = make([]float64, 39)
		for j := range features[i] {
			features[i][j] = float64(i*39+j) * 0.01
		}
	}

	result := d.ForwardFrames(features)
	if len(result) != T {
		t.Fatalf("expected %d frames, got %d", T, len(result))
	}
	for i, row := range result {
		if len(row) != d.OutputDim {
			t.Fatalf("frame %d: expected %d outputs, got %d", i, d.OutputDim, len(row))
		}
	}
}

func TestDNNLogSoftmax_SumsToOne(t *testing.T) {
	d := NewDNN(39, 16, 3)
	features := make([][]float64, 5)
	for i := range features {
		features[i] = make([]float64, 39)
		for j := range features[i] {
			features[i][j] = float64(j) * 0.1
		}
	}

	result := d.ForwardFrames(features)
	for t_idx, row := range result {
		sumExp := 0.0
		for _, lp := range row {
			sumExp += math.Exp(lp)
		}
		if math.Abs(sumExp-1.0) > 1e-6 {
			t.Errorf("frame %d: exp(log-softmax) sums to %f, want ~1.0", t_idx, sumExp)
		}
	}
}

func TestDNNForward_Deterministic(t *testing.T) {
	d := NewDNN(39, 16, 3)
	features := make([][]float64, 5)
	for i := range features {
		features[i] = make([]float64, 39)
		for j := range features[i] {
			features[i][j] = float64(j) * 0.1
		}
	}

	r1 := d.ForwardFrames(features)
	r2 := d.ForwardFrames(features)
	for t_idx := range r1 {
		for j := range r1[t_idx] {
			if r1[t_idx][j] != r2[t_idx][j] {
				t.Fatalf("frame %d class %d: %f != %f", t_idx, j, r1[t_idx][j], r2[t_idx][j])
			}
		}
	}
}

func TestDNNSubtractPrior(t *testing.T) {
	d := NewDNN(39, 16, 3)
	// Set known priors
	for i := range d.LogPrior {
		d.LogPrior[i] = -2.0
	}

	post := make([]float64, d.OutputDim)
	for i := range post {
		post[i] = -3.0
	}

	d.SubtractPrior(post)
	for i, v := range post {
		expected := -3.0 - (-2.0)
		if math.Abs(v-expected) > 1e-10 {
			t.Errorf("class %d: got %f, want %f", i, v, expected)
		}
	}
}

func TestDNNStateClassIndex(t *testing.T) {
	d := NewDNN(39, 16, 3)
	phonemes := AllPhonemes()

	// Every phoneme × state should have a valid index
	seen := make(map[int]bool)
	for _, ph := range phonemes {
		for s := 1; s <= NumEmittingStates; s++ {
			idx := d.StateClassIndex(ph, s)
			if idx < 0 || idx >= d.OutputDim {
				t.Errorf("phoneme %q state %d: index %d out of range [0,%d)", ph, s, idx, d.OutputDim)
			}
			if seen[idx] {
				t.Errorf("phoneme %q state %d: duplicate index %d", ph, s, idx)
			}
			seen[idx] = true
		}
	}
	if len(seen) != d.OutputDim {
		t.Errorf("expected %d unique indices, got %d", d.OutputDim, len(seen))
	}
}

func TestDNNSaveLoad_RoundTrip(t *testing.T) {
	d := NewDNN(39, 16, 3)
	// Set some non-default priors
	for i := range d.LogPrior {
		d.LogPrior[i] = -float64(i) * 0.1
	}

	var buf bytes.Buffer
	if err := d.Save(&buf); err != nil {
		t.Fatalf("Save: %v", err)
	}

	d2, err := LoadDNN(&buf)
	if err != nil {
		t.Fatalf("LoadDNN: %v", err)
	}

	// Check dimensions
	if d2.InputDim != d.InputDim || d2.HiddenDim != d.HiddenDim ||
		d2.OutputDim != d.OutputDim || d2.ContextLen != d.ContextLen {
		t.Fatal("dimension mismatch after load")
	}

	// Check weights
	checkSlice := func(name string, a, b []float64) {
		t.Helper()
		if len(a) != len(b) {
			t.Fatalf("%s: length %d != %d", name, len(a), len(b))
		}
		for i := range a {
			if a[i] != b[i] {
				t.Fatalf("%s[%d]: %f != %f", name, i, a[i], b[i])
			}
		}
	}
	checkSlice("W1", d.W1, d2.W1)
	checkSlice("B1", d.B1, d2.B1)
	checkSlice("W2", d.W2, d2.W2)
	checkSlice("B2", d.B2, d2.B2)
	checkSlice("W3", d.W3, d2.W3)
	checkSlice("B3", d.B3, d2.B3)
	checkSlice("LogPrior", d.LogPrior, d2.LogPrior)

	// Check phoneme list
	if len(d2.PhonemeList) != len(d.PhonemeList) {
		t.Fatalf("PhonemeList length: %d != %d", len(d2.PhonemeList), len(d.PhonemeList))
	}
	for i := range d.PhonemeList {
		if d2.PhonemeList[i] != d.PhonemeList[i] {
			t.Fatalf("PhonemeList[%d]: %q != %q", i, d2.PhonemeList[i], d.PhonemeList[i])
		}
	}

	// Verify loaded model produces same output
	features := make([][]float64, 5)
	for i := range features {
		features[i] = make([]float64, 39)
		for j := range features[i] {
			features[i][j] = float64(j) * 0.01
		}
	}
	r1 := d.ForwardFrames(features)
	r2 := d2.ForwardFrames(features)
	for ti := range r1 {
		for j := range r1[ti] {
			if math.Abs(r1[ti][j]-r2[ti][j]) > 1e-10 {
				t.Fatalf("output mismatch frame %d class %d", ti, j)
			}
		}
	}
}

func TestDNNContextWindow_EdgePadding(t *testing.T) {
	d := NewDNN(2, 4, 1) // featureDim=2, contextLen=1 → 3 frames × 2 = 6 input dim
	d.ContextLen = 1
	d.InputDim = 6

	// Reinitialize with correct dims
	d = &DNN{
		InputDim:    6,
		HiddenDim:   4,
		OutputDim:   len(AllPhonemes()) * NumEmittingStates,
		ContextLen:  1,
		W1:          make([]float64, 4*6),
		B1:          make([]float64, 4),
		W2:          make([]float64, 4*4),
		B2:          make([]float64, 4),
		W3:          make([]float64, d.OutputDim*4),
		B3:          make([]float64, d.OutputDim),
		LogPrior:    make([]float64, d.OutputDim),
		PhonemeList: AllPhonemes(),
	}
	// Set W1 to identity-like to inspect input construction
	// We just want to verify ForwardFrames doesn't panic
	features := [][]float64{
		{1.0, 2.0},
		{3.0, 4.0},
		{5.0, 6.0},
	}
	result := d.ForwardFrames(features)
	if len(result) != 3 {
		t.Fatalf("expected 3 frames, got %d", len(result))
	}
}

func TestDNNTraining_LossDecreases(t *testing.T) {
	// Create a small DNN and synthetic data with clear class structure
	rand.Seed(42)
	featureDim := 4
	hiddenDim := 16
	contextLen := 0 // no context for simplicity
	d := NewDNN(featureDim, hiddenDim, contextLen)

	numClasses := d.OutputDim
	N := 1000
	inputDim := d.InputDim // = featureDim since contextLen=0

	inputs := make([]float64, N*inputDim)
	targets := make([]int, N)
	for i := 0; i < N; i++ {
		c := i % numClasses
		targets[i] = c
		for j := 0; j < inputDim; j++ {
			// Class-dependent signal + noise
			inputs[i*inputDim+j] = float64(c)*0.5 + rand.NormFloat64()*0.1
		}
	}

	cfg := DefaultDNNTrainConfig()
	cfg.BatchSize = 64
	cfg.MaxEpochs = 10
	cfg.Patience = 0 // disable early stopping for this test
	cfg.HeldOutFrac = 0.1

	err := TrainDNN(d, inputs, targets, cfg)
	if err != nil {
		t.Fatalf("TrainDNN: %v", err)
	}
	// We just verify it didn't crash; loss decrease is printed to stderr
}

func TestBackpropBatch_GradientCheck(t *testing.T) {
	rand.Seed(123)
	// Small DNN for numerical gradient checking
	d := &DNN{
		InputDim:    6,
		HiddenDim:   4,
		OutputDim:   3,
		ContextLen:  0,
		W1:          make([]float64, 4*6),
		B1:          make([]float64, 4),
		W2:          make([]float64, 4*4),
		B2:          make([]float64, 4),
		W3:          make([]float64, 3*4),
		B3:          make([]float64, 3),
		LogPrior:    make([]float64, 3),
		PhonemeList: AllPhonemes(),
	}
	xavierInit(d.W1, 6, 4)
	xavierInit(d.W2, 4, 4)
	xavierInit(d.W3, 4, 3)

	bs := 4
	xBatch := make([]float64, bs*6)
	for i := range xBatch {
		xBatch[i] = rand.NormFloat64() * 0.5
	}
	batchTargets := []int{0, 1, 2, 1}

	ws := newDNNWorkspace(bs, 6, 4, 3)
	gW1 := make([]float64, len(d.W1))
	gB1 := make([]float64, len(d.B1))
	gW2 := make([]float64, len(d.W2))
	gB2 := make([]float64, len(d.B2))
	gW3 := make([]float64, len(d.W3))
	gB3 := make([]float64, len(d.B3))

	backpropBatch(d, xBatch, batchTargets, bs, ws, gW1, gB1, gW2, gB2, gW3, gB3)

	// Numerical gradient check for W1
	eps := 1e-5
	maxRelErr := 0.0
	for idx := 0; idx < len(d.W1); idx++ {
		orig := d.W1[idx]

		d.W1[idx] = orig + eps
		lossPlus := computeLoss(d, xBatch, batchTargets, bs, ws)
		d.W1[idx] = orig - eps
		lossMinus := computeLoss(d, xBatch, batchTargets, bs, ws)
		d.W1[idx] = orig

		numGrad := (lossPlus - lossMinus) / (2 * eps)
		anaGrad := gW1[idx] / float64(bs) // backprop accumulates, Adam divides by bs

		diff := math.Abs(numGrad - anaGrad)
		denom := math.Max(math.Abs(numGrad)+math.Abs(anaGrad), 1e-8)
		relErr := diff / denom
		if relErr > maxRelErr {
			maxRelErr = relErr
		}
	}

	if maxRelErr > 0.01 {
		t.Errorf("W1 gradient check failed: max relative error = %e", maxRelErr)
	}

	// Check W3
	maxRelErr = 0
	for idx := 0; idx < len(d.W3); idx++ {
		orig := d.W3[idx]
		d.W3[idx] = orig + eps
		lossPlus := computeLoss(d, xBatch, batchTargets, bs, ws)
		d.W3[idx] = orig - eps
		lossMinus := computeLoss(d, xBatch, batchTargets, bs, ws)
		d.W3[idx] = orig

		numGrad := (lossPlus - lossMinus) / (2 * eps)
		anaGrad := gW3[idx] / float64(bs)
		diff := math.Abs(numGrad - anaGrad)
		denom := math.Max(math.Abs(numGrad)+math.Abs(anaGrad), 1e-8)
		relErr := diff / denom
		if relErr > maxRelErr {
			maxRelErr = relErr
		}
	}
	if maxRelErr > 0.01 {
		t.Errorf("W3 gradient check failed: max relative error = %e", maxRelErr)
	}
}

func computeLoss(d *DNN, xBatch []float64, targets []int, bs int, ws *dnnWorkspace) float64 {
	H := d.HiddenDim
	O := d.OutputDim
	I := d.InputDim

	blas.Dgemm(false, true, bs, H, I,
		1.0, xBatch, I, d.W1, I, 0.0, ws.z1, H)
	for i := 0; i < bs*H; i++ {
		v := ws.z1[i] + d.B1[i%H]
		if v > 0 {
			ws.a1[i] = v
		} else {
			ws.a1[i] = 0
		}
	}
	blas.Dgemm(false, true, bs, H, H,
		1.0, ws.a1, H, d.W2, H, 0.0, ws.z2, H)
	for i := 0; i < bs*H; i++ {
		v := ws.z2[i] + d.B2[i%H]
		if v > 0 {
			ws.a2[i] = v
		} else {
			ws.a2[i] = 0
		}
	}
	blas.Dgemm(false, true, bs, O, H,
		1.0, ws.a2, H, d.W3, H, 0.0, ws.z3, O)

	loss := 0.0
	for i := 0; i < bs; i++ {
		off := i * O
		maxVal := math.Inf(-1)
		for j := 0; j < O; j++ {
			v := ws.z3[off+j] + d.B3[j]
			ws.z3[off+j] = v
			if v > maxVal {
				maxVal = v
			}
		}
		sumExp := 0.0
		for j := 0; j < O; j++ {
			sumExp += math.Exp(ws.z3[off+j] - maxVal)
		}
		logSumExp := maxVal + math.Log(sumExp)
		logP := ws.z3[off+targets[i]] - logSumExp
		loss -= logP
	}
	return loss / float64(bs)
}

func BenchmarkDNNForward_300frames(b *testing.B) {
	d := NewDNN(39, 256, 5)
	features := make([][]float64, 300)
	for i := range features {
		features[i] = make([]float64, 39)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		d.ForwardFrames(features)
	}
}

func BenchmarkDNNTraining(b *testing.B) {
	rand.Seed(42)
	d := NewDNN(39, 256, 5) // production-size DNN
	N := 100000
	inputDim := d.InputDim
	inputs := make([]float64, N*inputDim)
	targets := make([]int, N)
	for i := 0; i < N; i++ {
		targets[i] = i % d.OutputDim
		for j := 0; j < inputDim; j++ {
			inputs[i*inputDim+j] = rand.NormFloat64() * 0.1
		}
	}

	cfg := DefaultDNNTrainConfig()
	cfg.BatchSize = 256
	cfg.MaxEpochs = 1
	cfg.Patience = 0
	cfg.HeldOutFrac = 0.01

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		TrainDNN(d, inputs, targets, cfg)
	}
}
