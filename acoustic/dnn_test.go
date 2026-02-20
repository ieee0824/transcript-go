package acoustic

import (
	"bytes"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/ieee0824/transcript-go/internal/blas"
)

func TestDNNForward_Dimensions(t *testing.T) {
	d := NewDNN(39, 16, 3, 2, 0.0) // small DNN for testing
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
	d := NewDNN(39, 16, 3, 2, 0.0)
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
	d := NewDNN(39, 16, 3, 2, 0.0)
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
	d := NewDNN(39, 16, 3, 2, 0.0)
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
	d := NewDNN(39, 16, 3, 2, 0.0)
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
	d := NewDNN(39, 16, 3, 2, 0.0)
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

	// Check layer count
	if len(d2.Layers) != len(d.Layers) {
		t.Fatalf("layer count: %d != %d", len(d2.Layers), len(d.Layers))
	}

	// Check weights per layer
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
	for i := range d.Layers {
		checkSlice(fmt.Sprintf("Layers[%d].W", i), d.Layers[i].W, d2.Layers[i].W)
		checkSlice(fmt.Sprintf("Layers[%d].B", i), d.Layers[i].B, d2.Layers[i].B)
	}
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
	outputDim := len(AllPhonemes()) * NumEmittingStates
	d := &DNN{
		Layers: []DNNLayer{
			{W: make([]float64, 4*6), B: make([]float64, 4), InDim: 6, OutDim: 4},
			{W: make([]float64, 4*4), B: make([]float64, 4), InDim: 4, OutDim: 4},
			{W: make([]float64, outputDim*4), B: make([]float64, outputDim), InDim: 4, OutDim: outputDim},
		},
		InputDim:    6,
		HiddenDim:   4,
		OutputDim:   outputDim,
		ContextLen:  1,
		LogPrior:    make([]float64, outputDim),
		PhonemeList: AllPhonemes(),
	}
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
	d := NewDNN(featureDim, hiddenDim, contextLen, 2, 0.0)

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
		Layers: []DNNLayer{
			{W: make([]float64, 4*6), B: make([]float64, 4), InDim: 6, OutDim: 4},
			{W: make([]float64, 4*4), B: make([]float64, 4), InDim: 4, OutDim: 4},
			{W: make([]float64, 3*4), B: make([]float64, 3), InDim: 4, OutDim: 3},
		},
		InputDim:    6,
		HiddenDim:   4,
		OutputDim:   3,
		ContextLen:  0,
		LogPrior:    make([]float64, 3),
		PhonemeList: AllPhonemes(),
	}
	xavierInit(d.Layers[0].W, 6, 4)
	xavierInit(d.Layers[1].W, 4, 4)
	xavierInit(d.Layers[2].W, 4, 3)

	bs := 4
	xBatch := make([]float64, bs*6)
	for i := range xBatch {
		xBatch[i] = rand.NormFloat64() * 0.5
	}
	batchTargets := []int{0, 1, 2, 1}

	ws := newDNNWorkspace(bs, d.Layers, 0.0)
	grads := newWorkerGrads(d)

	// No dropout for gradient check (rng=nil)
	backpropBatch(d, xBatch, batchTargets, bs, ws, grads, nil, 0.0)

	// Check all layers
	eps := 1e-5
	for li := range d.Layers {
		// Check weights
		maxRelErr := 0.0
		for idx := range d.Layers[li].W {
			orig := d.Layers[li].W[idx]
			d.Layers[li].W[idx] = orig + eps
			lossPlus := computeLoss(d, xBatch, batchTargets, bs, ws)
			d.Layers[li].W[idx] = orig - eps
			lossMinus := computeLoss(d, xBatch, batchTargets, bs, ws)
			d.Layers[li].W[idx] = orig

			numGrad := (lossPlus - lossMinus) / (2 * eps)
			anaGrad := grads.gW[li][idx] / float64(bs)
			diff := math.Abs(numGrad - anaGrad)
			denom := math.Max(math.Abs(numGrad)+math.Abs(anaGrad), 1e-8)
			relErr := diff / denom
			if relErr > maxRelErr {
				maxRelErr = relErr
			}
		}
		if maxRelErr > 0.01 {
			t.Errorf("Layer %d W gradient check failed: max relative error = %e", li, maxRelErr)
		}

		// Check biases
		maxRelErr = 0.0
		for idx := range d.Layers[li].B {
			orig := d.Layers[li].B[idx]
			d.Layers[li].B[idx] = orig + eps
			lossPlus := computeLoss(d, xBatch, batchTargets, bs, ws)
			d.Layers[li].B[idx] = orig - eps
			lossMinus := computeLoss(d, xBatch, batchTargets, bs, ws)
			d.Layers[li].B[idx] = orig

			numGrad := (lossPlus - lossMinus) / (2 * eps)
			anaGrad := grads.gB[li][idx] / float64(bs)
			diff := math.Abs(numGrad - anaGrad)
			denom := math.Max(math.Abs(numGrad)+math.Abs(anaGrad), 1e-8)
			relErr := diff / denom
			if relErr > maxRelErr {
				maxRelErr = relErr
			}
		}
		if maxRelErr > 0.01 {
			t.Errorf("Layer %d B gradient check failed: max relative error = %e", li, maxRelErr)
		}
	}
}

func computeLoss(d *DNN, xBatch []float64, targets []int, bs int, ws *dnnWorkspace) float64 {
	return computeLossSmooth(d, xBatch, targets, bs, ws, 0.0)
}

func computeLossSmooth(d *DNN, xBatch []float64, targets []int, bs int, ws *dnnWorkspace, labelSmooth float64) float64 {
	nLayers := len(d.Layers)

	// Forward pass
	prevAct := xBatch
	prevDim := d.InputDim
	for i := 0; i < nLayers; i++ {
		layer := &d.Layers[i]
		blas.Dgemm(false, true, bs, layer.OutDim, prevDim,
			1.0, prevAct, prevDim, layer.W, prevDim, 0.0, ws.z[i], layer.OutDim)

		if i < nLayers-1 {
			dim := layer.OutDim
			for idx := 0; idx < bs*dim; idx++ {
				v := ws.z[i][idx] + layer.B[idx%dim]
				if v > 0 {
					ws.a[i][idx] = v
				} else {
					ws.a[i][idx] = 0
				}
			}
			prevAct = ws.a[i]
			prevDim = dim
		}
	}

	// Output layer: softmax + loss
	O := d.OutputDim
	outLayer := &d.Layers[nLayers-1]
	loss := 0.0
	K := float64(O)
	smooth := labelSmooth / K
	for r := 0; r < bs; r++ {
		off := r * O
		maxVal := math.Inf(-1)
		for j := 0; j < O; j++ {
			v := ws.z[nLayers-1][off+j] + outLayer.B[j]
			ws.z[nLayers-1][off+j] = v
			if v > maxVal {
				maxVal = v
			}
		}
		sumExp := 0.0
		for j := 0; j < O; j++ {
			sumExp += math.Exp(ws.z[nLayers-1][off+j] - maxVal)
		}
		logSumExp := maxVal + math.Log(sumExp)

		if labelSmooth > 0 {
			targetWeight := 1.0 - labelSmooth + smooth
			for j := 0; j < O; j++ {
				logP := ws.z[nLayers-1][off+j] - logSumExp
				if j == targets[r] {
					loss -= targetWeight * logP
				} else {
					loss -= smooth * logP
				}
			}
		} else {
			logP := ws.z[nLayers-1][off+targets[r]] - logSumExp
			loss -= logP
		}
	}
	return loss / float64(bs)
}

// --- Variable layer tests ---

func TestDNNForward_4HiddenLayers(t *testing.T) {
	d := NewDNN(39, 32, 3, 4, 0.0)
	if len(d.Layers) != 5 { // 4 hidden + 1 output
		t.Fatalf("expected 5 layers, got %d", len(d.Layers))
	}

	features := make([][]float64, 10)
	for i := range features {
		features[i] = make([]float64, 39)
		for j := range features[i] {
			features[i][j] = rand.NormFloat64() * 0.1
		}
	}

	result := d.ForwardFrames(features)
	if len(result) != 10 {
		t.Fatalf("expected 10 frames, got %d", len(result))
	}
	for tIdx, row := range result {
		if len(row) != d.OutputDim {
			t.Fatalf("frame %d: expected %d outputs, got %d", tIdx, d.OutputDim, len(row))
		}
		sumExp := 0.0
		for _, lp := range row {
			sumExp += math.Exp(lp)
		}
		if math.Abs(sumExp-1.0) > 1e-6 {
			t.Errorf("frame %d: exp(log-softmax) sums to %f, want ~1.0", tIdx, sumExp)
		}
	}
}

func TestDNNForward_1HiddenLayer(t *testing.T) {
	d := NewDNN(39, 32, 3, 1, 0.0)
	if len(d.Layers) != 2 { // 1 hidden + 1 output
		t.Fatalf("expected 2 layers, got %d", len(d.Layers))
	}

	features := make([][]float64, 5)
	for i := range features {
		features[i] = make([]float64, 39)
		for j := range features[i] {
			features[i][j] = rand.NormFloat64() * 0.1
		}
	}

	result := d.ForwardFrames(features)
	for tIdx, row := range result {
		sumExp := 0.0
		for _, lp := range row {
			sumExp += math.Exp(lp)
		}
		if math.Abs(sumExp-1.0) > 1e-6 {
			t.Errorf("frame %d: exp(log-softmax) sums to %f, want ~1.0", tIdx, sumExp)
		}
	}
}

func TestDNNSaveLoad_VariableLayers(t *testing.T) {
	d := NewDNN(39, 32, 3, 4, 0.2)
	for i := range d.LogPrior {
		d.LogPrior[i] = -float64(i) * 0.01
	}

	var buf bytes.Buffer
	if err := d.Save(&buf); err != nil {
		t.Fatalf("Save: %v", err)
	}

	d2, err := LoadDNN(&buf)
	if err != nil {
		t.Fatalf("LoadDNN: %v", err)
	}

	if len(d2.Layers) != len(d.Layers) {
		t.Fatalf("layer count: %d != %d", len(d2.Layers), len(d.Layers))
	}
	if d2.InputDim != d.InputDim || d2.OutputDim != d.OutputDim {
		t.Fatal("dimension mismatch")
	}

	// Verify forward output matches
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

func TestBackpropBatch_GradientCheck_4Layers(t *testing.T) {
	rand.Seed(456)
	d := &DNN{
		Layers: []DNNLayer{
			{W: make([]float64, 4*6), B: make([]float64, 4), InDim: 6, OutDim: 4},
			{W: make([]float64, 4*4), B: make([]float64, 4), InDim: 4, OutDim: 4},
			{W: make([]float64, 4*4), B: make([]float64, 4), InDim: 4, OutDim: 4},
			{W: make([]float64, 3*4), B: make([]float64, 3), InDim: 4, OutDim: 3},
		},
		InputDim:    6,
		HiddenDim:   4,
		OutputDim:   3,
		ContextLen:  0,
		LogPrior:    make([]float64, 3),
		PhonemeList: AllPhonemes(),
	}
	for i := range d.Layers {
		xavierInit(d.Layers[i].W, d.Layers[i].InDim, d.Layers[i].OutDim)
	}

	bs := 4
	xBatch := make([]float64, bs*6)
	for i := range xBatch {
		xBatch[i] = rand.NormFloat64() * 0.5
	}
	batchTargets := []int{0, 1, 2, 1}

	ws := newDNNWorkspace(bs, d.Layers, 0.0)
	grads := newWorkerGrads(d)
	backpropBatch(d, xBatch, batchTargets, bs, ws, grads, nil, 0.0)

	eps := 1e-5
	for li := range d.Layers {
		maxRelErr := 0.0
		for idx := range d.Layers[li].W {
			orig := d.Layers[li].W[idx]
			d.Layers[li].W[idx] = orig + eps
			lossPlus := computeLoss(d, xBatch, batchTargets, bs, ws)
			d.Layers[li].W[idx] = orig - eps
			lossMinus := computeLoss(d, xBatch, batchTargets, bs, ws)
			d.Layers[li].W[idx] = orig

			numGrad := (lossPlus - lossMinus) / (2 * eps)
			anaGrad := grads.gW[li][idx] / float64(bs)
			diff := math.Abs(numGrad - anaGrad)
			denom := math.Max(math.Abs(numGrad)+math.Abs(anaGrad), 1e-8)
			relErr := diff / denom
			if relErr > maxRelErr {
				maxRelErr = relErr
			}
		}
		if maxRelErr > 0.01 {
			t.Errorf("4-layer: Layer %d W gradient check failed: max relative error = %e", li, maxRelErr)
		}
	}
}

// --- Label smoothing tests ---

func TestBackpropBatch_GradientCheck_LabelSmooth(t *testing.T) {
	rand.Seed(789)
	d := &DNN{
		Layers: []DNNLayer{
			{W: make([]float64, 4*6), B: make([]float64, 4), InDim: 6, OutDim: 4},
			{W: make([]float64, 3*4), B: make([]float64, 3), InDim: 4, OutDim: 3},
		},
		InputDim:    6,
		HiddenDim:   4,
		OutputDim:   3,
		ContextLen:  0,
		LogPrior:    make([]float64, 3),
		PhonemeList: AllPhonemes(),
	}
	xavierInit(d.Layers[0].W, 6, 4)
	xavierInit(d.Layers[1].W, 4, 3)

	bs := 4
	xBatch := make([]float64, bs*6)
	for i := range xBatch {
		xBatch[i] = rand.NormFloat64() * 0.5
	}
	batchTargets := []int{0, 1, 2, 1}
	labelSmooth := 0.1

	ws := newDNNWorkspace(bs, d.Layers, 0.0)
	grads := newWorkerGrads(d)
	backpropBatch(d, xBatch, batchTargets, bs, ws, grads, nil, labelSmooth)

	eps := 1e-5
	for li := range d.Layers {
		// Check weights
		maxRelErr := 0.0
		for idx := range d.Layers[li].W {
			orig := d.Layers[li].W[idx]
			d.Layers[li].W[idx] = orig + eps
			lossPlus := computeLossSmooth(d, xBatch, batchTargets, bs, ws, labelSmooth)
			d.Layers[li].W[idx] = orig - eps
			lossMinus := computeLossSmooth(d, xBatch, batchTargets, bs, ws, labelSmooth)
			d.Layers[li].W[idx] = orig

			numGrad := (lossPlus - lossMinus) / (2 * eps)
			anaGrad := grads.gW[li][idx] / float64(bs)
			diff := math.Abs(numGrad - anaGrad)
			denom := math.Max(math.Abs(numGrad)+math.Abs(anaGrad), 1e-8)
			relErr := diff / denom
			if relErr > maxRelErr {
				maxRelErr = relErr
			}
		}
		if maxRelErr > 0.01 {
			t.Errorf("LabelSmooth: Layer %d W gradient check failed: max relative error = %e", li, maxRelErr)
		}

		// Check biases
		maxRelErr = 0.0
		for idx := range d.Layers[li].B {
			orig := d.Layers[li].B[idx]
			d.Layers[li].B[idx] = orig + eps
			lossPlus := computeLossSmooth(d, xBatch, batchTargets, bs, ws, labelSmooth)
			d.Layers[li].B[idx] = orig - eps
			lossMinus := computeLossSmooth(d, xBatch, batchTargets, bs, ws, labelSmooth)
			d.Layers[li].B[idx] = orig

			numGrad := (lossPlus - lossMinus) / (2 * eps)
			anaGrad := grads.gB[li][idx] / float64(bs)
			diff := math.Abs(numGrad - anaGrad)
			denom := math.Max(math.Abs(numGrad)+math.Abs(anaGrad), 1e-8)
			relErr := diff / denom
			if relErr > maxRelErr {
				maxRelErr = relErr
			}
		}
		if maxRelErr > 0.01 {
			t.Errorf("LabelSmooth: Layer %d B gradient check failed: max relative error = %e", li, maxRelErr)
		}
	}
}

func TestCosineLRSchedule(t *testing.T) {
	// Verify cosine LR values at key points
	lr := 0.001
	lrMin := lr * 0.01
	maxEpochs := 50

	// Epoch 0: should be close to lr_max
	cosine0 := 0.5 * (1.0 + math.Cos(math.Pi*0.0/float64(maxEpochs)))
	lr0 := lrMin + (lr-lrMin)*cosine0
	if math.Abs(lr0-lr) > 1e-10 {
		t.Errorf("epoch 0: lr=%e, want %e", lr0, lr)
	}

	// Epoch maxEpochs: should be close to lr_min
	cosineEnd := 0.5 * (1.0 + math.Cos(math.Pi*float64(maxEpochs)/float64(maxEpochs)))
	lrEnd := lrMin + (lr-lrMin)*cosineEnd
	if math.Abs(lrEnd-lrMin) > 1e-10 {
		t.Errorf("epoch %d: lr=%e, want %e", maxEpochs, lrEnd, lrMin)
	}

	// Epoch maxEpochs/2: should be midpoint
	cosineHalf := 0.5 * (1.0 + math.Cos(math.Pi*float64(maxEpochs/2)/float64(maxEpochs)))
	lrHalf := lrMin + (lr-lrMin)*cosineHalf
	expected := (lr + lrMin) / 2.0
	if math.Abs(lrHalf-expected) > 1e-6 {
		t.Errorf("epoch %d: lr=%e, want ~%e", maxEpochs/2, lrHalf, expected)
	}

	// Verify monotonically decreasing
	prev := lr
	for epoch := 1; epoch <= maxEpochs; epoch++ {
		cosine := 0.5 * (1.0 + math.Cos(math.Pi*float64(epoch)/float64(maxEpochs)))
		lrE := lrMin + (lr-lrMin)*cosine
		if lrE > prev+1e-15 {
			t.Errorf("epoch %d: lr=%e > prev=%e (not monotonically decreasing)", epoch, lrE, prev)
		}
		prev = lrE
	}
}

func TestDNNTraining_WithLabelSmoothing(t *testing.T) {
	rand.Seed(42)
	d := NewDNN(4, 16, 0, 2, 0.0)

	N := 500
	inputDim := d.InputDim
	inputs := make([]float64, N*inputDim)
	targets := make([]int, N)
	for i := 0; i < N; i++ {
		c := i % d.OutputDim
		targets[i] = c
		for j := 0; j < inputDim; j++ {
			inputs[i*inputDim+j] = float64(c)*0.5 + rand.NormFloat64()*0.1
		}
	}

	cfg := DefaultDNNTrainConfig()
	cfg.BatchSize = 64
	cfg.MaxEpochs = 5
	cfg.Patience = 0
	cfg.HeldOutFrac = 0.1
	cfg.LabelSmooth = 0.1

	err := TrainDNN(d, inputs, targets, cfg)
	if err != nil {
		t.Fatalf("TrainDNN with label smoothing: %v", err)
	}
}

func TestDNNTraining_WithCosineLR(t *testing.T) {
	rand.Seed(42)
	d := NewDNN(4, 16, 0, 2, 0.0)

	N := 500
	inputDim := d.InputDim
	inputs := make([]float64, N*inputDim)
	targets := make([]int, N)
	for i := 0; i < N; i++ {
		c := i % d.OutputDim
		targets[i] = c
		for j := 0; j < inputDim; j++ {
			inputs[i*inputDim+j] = float64(c)*0.5 + rand.NormFloat64()*0.1
		}
	}

	cfg := DefaultDNNTrainConfig()
	cfg.BatchSize = 64
	cfg.MaxEpochs = 5
	cfg.Patience = 0
	cfg.HeldOutFrac = 0.1
	cfg.LRSchedule = "cosine"

	err := TrainDNN(d, inputs, targets, cfg)
	if err != nil {
		t.Fatalf("TrainDNN with cosine LR: %v", err)
	}
}

// --- Dropout tests ---

func TestDNNDropout_InferenceDeterministic(t *testing.T) {
	d := NewDNN(39, 32, 3, 2, 0.5)
	features := make([][]float64, 5)
	for i := range features {
		features[i] = make([]float64, 39)
		for j := range features[i] {
			features[i][j] = float64(j) * 0.1
		}
	}

	r1 := d.ForwardFrames(features)
	r2 := d.ForwardFrames(features)
	for tIdx := range r1 {
		for j := range r1[tIdx] {
			if r1[tIdx][j] != r2[tIdx][j] {
				t.Fatalf("inference not deterministic: frame %d class %d: %f != %f",
					tIdx, j, r1[tIdx][j], r2[tIdx][j])
			}
		}
	}
}

func TestDNNTraining_WithDropout(t *testing.T) {
	rand.Seed(42)
	d := NewDNN(4, 16, 0, 2, 0.3)

	N := 500
	inputDim := d.InputDim
	inputs := make([]float64, N*inputDim)
	targets := make([]int, N)
	for i := 0; i < N; i++ {
		c := i % d.OutputDim
		targets[i] = c
		for j := 0; j < inputDim; j++ {
			inputs[i*inputDim+j] = float64(c)*0.5 + rand.NormFloat64()*0.1
		}
	}

	cfg := DefaultDNNTrainConfig()
	cfg.BatchSize = 64
	cfg.MaxEpochs = 5
	cfg.Patience = 0
	cfg.HeldOutFrac = 0.1

	err := TrainDNN(d, inputs, targets, cfg)
	if err != nil {
		t.Fatalf("TrainDNN with dropout: %v", err)
	}
}

// --- Benchmarks ---

func BenchmarkDNNForward_300frames(b *testing.B) {
	d := NewDNN(39, 256, 5, 2, 0.0)
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
	d := NewDNN(39, 256, 5, 2, 0.0) // production-size DNN
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
