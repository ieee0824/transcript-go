package acoustic

import (
	"fmt"
	"math"
	"math/rand"
	"os"

	"github.com/ieee0824/transcript-go/internal/blas"
)

// DNNTrainConfig holds DNN training hyperparameters.
type DNNTrainConfig struct {
	LearningRate float64
	Beta1        float64 // Adam beta1
	Beta2        float64 // Adam beta2
	Epsilon      float64 // Adam epsilon
	BatchSize    int
	MaxEpochs    int
	Patience     int     // early stopping patience (0 = disabled)
	HeldOutFrac  float64 // fraction held out for validation
}

// DefaultDNNTrainConfig returns sensible defaults for DNN training.
func DefaultDNNTrainConfig() DNNTrainConfig {
	return DNNTrainConfig{
		LearningRate: 0.001,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
		BatchSize:    256,
		MaxEpochs:    20,
		Patience:     3,
		HeldOutFrac:  0.1,
	}
}

// dnnWorkspace holds pre-allocated buffers for one mini-batch forward/backward pass.
type dnnWorkspace struct {
	batchSize int
	// Forward intermediates
	xBatch []float64 // [batchSize × InputDim]
	z1     []float64 // [batchSize × HiddenDim] pre-activation layer 1
	a1     []float64 // [batchSize × HiddenDim] post-activation layer 1
	z2     []float64 // [batchSize × HiddenDim]
	a2     []float64 // [batchSize × HiddenDim]
	z3     []float64 // [batchSize × OutputDim]
	prob   []float64 // [batchSize × OutputDim] softmax probabilities

	// Backward intermediates
	dz3 []float64 // [batchSize × OutputDim]
	da2 []float64 // [batchSize × HiddenDim]
	dz2 []float64 // [batchSize × HiddenDim]
	da1 []float64 // [batchSize × HiddenDim]
	dz1 []float64 // [batchSize × HiddenDim]
}

func newDNNWorkspace(batchSize, inputDim, hiddenDim, outputDim int) *dnnWorkspace {
	return &dnnWorkspace{
		batchSize: batchSize,
		xBatch:    make([]float64, batchSize*inputDim),
		z1:        make([]float64, batchSize*hiddenDim),
		a1:        make([]float64, batchSize*hiddenDim),
		z2:        make([]float64, batchSize*hiddenDim),
		a2:        make([]float64, batchSize*hiddenDim),
		z3:        make([]float64, batchSize*outputDim),
		prob:      make([]float64, batchSize*outputDim),
		dz3:       make([]float64, batchSize*outputDim),
		da2:       make([]float64, batchSize*hiddenDim),
		dz2:       make([]float64, batchSize*hiddenDim),
		da1:       make([]float64, batchSize*hiddenDim),
		dz1:       make([]float64, batchSize*hiddenDim),
	}
}

// adamState holds per-parameter momentum and variance for Adam optimizer.
type adamState struct {
	mW1, vW1, mB1, vB1 []float64
	mW2, vW2, mB2, vB2 []float64
	mW3, vW3, mB3, vB3 []float64
	t                   int // step counter
}

func newAdamState(d *DNN) *adamState {
	return &adamState{
		mW1: make([]float64, len(d.W1)), vW1: make([]float64, len(d.W1)),
		mB1: make([]float64, len(d.B1)), vB1: make([]float64, len(d.B1)),
		mW2: make([]float64, len(d.W2)), vW2: make([]float64, len(d.W2)),
		mB2: make([]float64, len(d.B2)), vB2: make([]float64, len(d.B2)),
		mW3: make([]float64, len(d.W3)), vW3: make([]float64, len(d.W3)),
		mB3: make([]float64, len(d.B3)), vB3: make([]float64, len(d.B3)),
	}
}

// TrainDNN trains the DNN on (input, target) sample pairs with mini-batch Adam.
// inputs: flat [N × InputDim], targets: [N] class indices.
// Reports progress to stderr.
func TrainDNN(dnn *DNN, inputs []float64, targets []int, cfg DNNTrainConfig) error {
	N := len(targets)
	if N == 0 {
		return fmt.Errorf("no training samples")
	}

	// Split into train/validation
	valN := int(float64(N) * cfg.HeldOutFrac)
	if valN < 1 {
		valN = 1
	}
	trainN := N - valN

	// Shuffle indices
	perm := rand.Perm(N)
	trainIdx := perm[:trainN]
	valIdx := perm[trainN:]

	ws := newDNNWorkspace(cfg.BatchSize, dnn.InputDim, dnn.HiddenDim, dnn.OutputDim)
	adam := newAdamState(dnn)

	// Gradient buffers
	gW1 := make([]float64, len(dnn.W1))
	gB1 := make([]float64, len(dnn.B1))
	gW2 := make([]float64, len(dnn.W2))
	gB2 := make([]float64, len(dnn.B2))
	gW3 := make([]float64, len(dnn.W3))
	gB3 := make([]float64, len(dnn.B3))

	bestValLoss := math.Inf(1)
	patience := 0

	for epoch := 0; epoch < cfg.MaxEpochs; epoch++ {
		// Shuffle training indices
		rand.Shuffle(trainN, func(i, j int) {
			trainIdx[i], trainIdx[j] = trainIdx[j], trainIdx[i]
		})

		totalLoss := 0.0
		totalCorrect := 0
		nBatches := 0

		for start := 0; start < trainN; start += cfg.BatchSize {
			end := start + cfg.BatchSize
			if end > trainN {
				end = trainN
			}
			bs := end - start

			// Fill batch
			fillBatch(inputs, targets, trainIdx[start:end], dnn.InputDim, ws.xBatch)
			batchTargets := make([]int, bs)
			for i := 0; i < bs; i++ {
				batchTargets[i] = targets[trainIdx[start+i]]
			}

			loss, correct := backpropBatch(dnn, ws.xBatch, batchTargets, bs, ws,
				gW1, gB1, gW2, gB2, gW3, gB3)
			totalLoss += loss
			totalCorrect += correct
			nBatches++

			// Adam update
			invBS := 1.0 / float64(bs)
			adam.t++
			adamUpdate(dnn.W1, gW1, adam.mW1, adam.vW1, cfg.LearningRate, cfg.Beta1, cfg.Beta2, cfg.Epsilon, adam.t, invBS)
			adamUpdate(dnn.B1, gB1, adam.mB1, adam.vB1, cfg.LearningRate, cfg.Beta1, cfg.Beta2, cfg.Epsilon, adam.t, invBS)
			adamUpdate(dnn.W2, gW2, adam.mW2, adam.vW2, cfg.LearningRate, cfg.Beta1, cfg.Beta2, cfg.Epsilon, adam.t, invBS)
			adamUpdate(dnn.B2, gB2, adam.mB2, adam.vB2, cfg.LearningRate, cfg.Beta1, cfg.Beta2, cfg.Epsilon, adam.t, invBS)
			adamUpdate(dnn.W3, gW3, adam.mW3, adam.vW3, cfg.LearningRate, cfg.Beta1, cfg.Beta2, cfg.Epsilon, adam.t, invBS)
			adamUpdate(dnn.B3, gB3, adam.mB3, adam.vB3, cfg.LearningRate, cfg.Beta1, cfg.Beta2, cfg.Epsilon, adam.t, invBS)
		}

		trainLoss := totalLoss / float64(nBatches)
		trainAcc := float64(totalCorrect) / float64(trainN) * 100

		// Validation
		valLoss, valAcc := evaluateDNN(dnn, inputs, targets, valIdx, cfg.BatchSize, ws)

		fmt.Fprintf(os.Stderr, "  Epoch %2d: train_loss=%.4f train_acc=%.1f%% val_loss=%.4f val_acc=%.1f%%\n",
			epoch+1, trainLoss, trainAcc, valLoss, valAcc)

		// Early stopping
		if cfg.Patience > 0 {
			if valLoss < bestValLoss-1e-4 {
				bestValLoss = valLoss
				patience = 0
			} else {
				patience++
				if patience >= cfg.Patience {
					fmt.Fprintf(os.Stderr, "  Early stopping at epoch %d\n", epoch+1)
					break
				}
			}
		}
	}
	return nil
}

func fillBatch(inputs []float64, targets []int, indices []int, inputDim int, xBatch []float64) {
	for i, idx := range indices {
		copy(xBatch[i*inputDim:(i+1)*inputDim], inputs[idx*inputDim:(idx+1)*inputDim])
	}
}

// backpropBatch computes forward pass, loss, and gradients for one mini-batch.
// Returns average cross-entropy loss and number of correct predictions.
func backpropBatch(dnn *DNN, xBatch []float64, batchTargets []int, bs int,
	ws *dnnWorkspace,
	gW1, gB1, gW2, gB2, gW3, gB3 []float64) (float64, int) {

	I := dnn.InputDim
	H := dnn.HiddenDim
	O := dnn.OutputDim

	// === Forward pass ===

	// Layer 1: z1 = X @ W1^T, a1 = ReLU(z1 + b1)
	blas.Dgemm(false, true, bs, H, I,
		1.0, xBatch, I, dnn.W1, I, 0.0, ws.z1, H)
	for i := 0; i < bs; i++ {
		for j := 0; j < H; j++ {
			v := ws.z1[i*H+j] + dnn.B1[j]
			ws.z1[i*H+j] = v
			if v > 0 {
				ws.a1[i*H+j] = v
			} else {
				ws.a1[i*H+j] = 0
			}
		}
	}

	// Layer 2: z2 = a1 @ W2^T, a2 = ReLU(z2 + b2)
	blas.Dgemm(false, true, bs, H, H,
		1.0, ws.a1, H, dnn.W2, H, 0.0, ws.z2, H)
	for i := 0; i < bs; i++ {
		for j := 0; j < H; j++ {
			v := ws.z2[i*H+j] + dnn.B2[j]
			ws.z2[i*H+j] = v
			if v > 0 {
				ws.a2[i*H+j] = v
			} else {
				ws.a2[i*H+j] = 0
			}
		}
	}

	// Layer 3: z3 = a2 @ W3^T + b3, prob = softmax(z3)
	blas.Dgemm(false, true, bs, O, H,
		1.0, ws.a2, H, dnn.W3, H, 0.0, ws.z3, O)

	totalLoss := 0.0
	correct := 0
	for i := 0; i < bs; i++ {
		off := i * O
		// Add bias + softmax
		maxVal := math.Inf(-1)
		for j := 0; j < O; j++ {
			ws.z3[off+j] += dnn.B3[j]
			if ws.z3[off+j] > maxVal {
				maxVal = ws.z3[off+j]
			}
		}
		sumExp := 0.0
		for j := 0; j < O; j++ {
			ws.prob[off+j] = math.Exp(ws.z3[off+j] - maxVal)
			sumExp += ws.prob[off+j]
		}
		for j := 0; j < O; j++ {
			ws.prob[off+j] /= sumExp
		}

		// Cross-entropy loss
		t := batchTargets[i]
		p := ws.prob[off+t]
		if p < 1e-30 {
			p = 1e-30
		}
		totalLoss -= math.Log(p)

		// Accuracy
		bestJ := 0
		bestP := ws.prob[off]
		for j := 1; j < O; j++ {
			if ws.prob[off+j] > bestP {
				bestP = ws.prob[off+j]
				bestJ = j
			}
		}
		if bestJ == t {
			correct++
		}
	}

	// === Backward pass ===

	// dz3 = prob - one_hot(target) [softmax + CE gradient]
	for i := 0; i < bs*O; i++ {
		ws.dz3[i] = ws.prob[i]
	}
	for i := 0; i < bs; i++ {
		ws.dz3[i*O+batchTargets[i]] -= 1.0
	}

	// gW3 = dz3^T @ a2  [O×bs] × [bs×H] = [O×H]
	clearSlice(gW3)
	blas.Dgemm(true, false, O, H, bs,
		1.0, ws.dz3, O, ws.a2, H, 0.0, gW3, H)

	// gB3 = sum(dz3, axis=0)
	clearSlice(gB3)
	for i := 0; i < bs; i++ {
		for j := 0; j < O; j++ {
			gB3[j] += ws.dz3[i*O+j]
		}
	}

	// da2 = dz3 @ W3  [bs×O] × [O×H] = [bs×H]
	blas.Dgemm(false, false, bs, H, O,
		1.0, ws.dz3, O, dnn.W3, H, 0.0, ws.da2, H)

	// dz2 = da2 * ReLU'(z2)
	for i := 0; i < bs*H; i++ {
		if ws.z2[i] > 0 {
			ws.dz2[i] = ws.da2[i]
		} else {
			ws.dz2[i] = 0
		}
	}

	// gW2 = dz2^T @ a1
	clearSlice(gW2)
	blas.Dgemm(true, false, H, H, bs,
		1.0, ws.dz2, H, ws.a1, H, 0.0, gW2, H)

	// gB2 = sum(dz2, axis=0)
	clearSlice(gB2)
	for i := 0; i < bs; i++ {
		for j := 0; j < H; j++ {
			gB2[j] += ws.dz2[i*H+j]
		}
	}

	// da1 = dz2 @ W2
	blas.Dgemm(false, false, bs, H, H,
		1.0, ws.dz2, H, dnn.W2, H, 0.0, ws.da1, H)

	// dz1 = da1 * ReLU'(z1)
	for i := 0; i < bs*H; i++ {
		if ws.z1[i] > 0 {
			ws.dz1[i] = ws.da1[i]
		} else {
			ws.dz1[i] = 0
		}
	}

	// gW1 = dz1^T @ X
	clearSlice(gW1)
	blas.Dgemm(true, false, H, I, bs,
		1.0, ws.dz1, H, xBatch, I, 0.0, gW1, I)

	// gB1 = sum(dz1, axis=0)
	clearSlice(gB1)
	for i := 0; i < bs; i++ {
		for j := 0; j < H; j++ {
			gB1[j] += ws.dz1[i*H+j]
		}
	}

	return totalLoss / float64(bs), correct
}

// adamUpdate applies one Adam step: params -= lr * m_hat / (sqrt(v_hat) + eps)
// gradScale is applied to gradients (typically 1/batchSize).
func adamUpdate(params, grad, m, v []float64, lr, beta1, beta2, eps float64, t int, gradScale float64) {
	bc1 := 1.0 - math.Pow(beta1, float64(t))
	bc2 := 1.0 - math.Pow(beta2, float64(t))
	for i := range params {
		g := grad[i] * gradScale
		m[i] = beta1*m[i] + (1-beta1)*g
		v[i] = beta2*v[i] + (1-beta2)*g*g
		mHat := m[i] / bc1
		vHat := v[i] / bc2
		params[i] -= lr * mHat / (math.Sqrt(vHat) + eps)
	}
}

// evaluateDNN computes average loss and accuracy on a subset of data.
func evaluateDNN(dnn *DNN, inputs []float64, targets []int, indices []int, batchSize int, ws *dnnWorkspace) (float64, float64) {
	N := len(indices)
	if N == 0 {
		return 0, 0
	}

	I := dnn.InputDim
	O := dnn.OutputDim
	H := dnn.HiddenDim

	totalLoss := 0.0
	totalCorrect := 0

	for start := 0; start < N; start += batchSize {
		end := start + batchSize
		if end > N {
			end = N
		}
		bs := end - start

		fillBatch(inputs, targets, indices[start:end], I, ws.xBatch)

		// Forward only
		blas.Dgemm(false, true, bs, H, I,
			1.0, ws.xBatch, I, dnn.W1, I, 0.0, ws.z1, H)
		for i := 0; i < bs*H; i++ {
			v := ws.z1[i] + dnn.B1[i%H]
			if v > 0 {
				ws.a1[i] = v
			} else {
				ws.a1[i] = 0
			}
		}
		blas.Dgemm(false, true, bs, H, H,
			1.0, ws.a1, H, dnn.W2, H, 0.0, ws.z2, H)
		for i := 0; i < bs*H; i++ {
			v := ws.z2[i] + dnn.B2[i%H]
			if v > 0 {
				ws.a2[i] = v
			} else {
				ws.a2[i] = 0
			}
		}
		blas.Dgemm(false, true, bs, O, H,
			1.0, ws.a2, H, dnn.W3, H, 0.0, ws.z3, O)

		for i := 0; i < bs; i++ {
			off := i * O
			maxVal := math.Inf(-1)
			for j := 0; j < O; j++ {
				ws.z3[off+j] += dnn.B3[j]
				if ws.z3[off+j] > maxVal {
					maxVal = ws.z3[off+j]
				}
			}
			sumExp := 0.0
			for j := 0; j < O; j++ {
				e := math.Exp(ws.z3[off+j] - maxVal)
				ws.prob[off+j] = e
				sumExp += e
			}

			t := targets[indices[start+i]]
			p := ws.prob[off+t] / sumExp
			if p < 1e-30 {
				p = 1e-30
			}
			totalLoss -= math.Log(p)

			bestJ := 0
			bestP := ws.prob[off]
			for j := 1; j < O; j++ {
				if ws.prob[off+j] > bestP {
					bestP = ws.prob[off+j]
					bestJ = j
				}
			}
			if bestJ == t {
				totalCorrect++
			}
		}
	}

	return totalLoss / float64(N), float64(totalCorrect) / float64(N) * 100
}

func clearSlice(s []float64) {
	for i := range s {
		s[i] = 0
	}
}
