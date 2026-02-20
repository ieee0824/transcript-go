package acoustic

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sync"

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
	xBatch    []float64   // [batchSize × InputDim]
	z         [][]float64 // z[i] = pre-activation for layer i [batchSize × layer.OutDim]
	a         [][]float64 // a[i] = post-activation for hidden layer i [batchSize × layer.OutDim]
	prob      []float64   // [batchSize × OutputDim] softmax output
	masks     [][]float64 // dropout masks for hidden layers (nil if no dropout)

	// Backward intermediates
	dz [][]float64 // dz[i] for each layer
	da [][]float64 // da[i] for each hidden layer
}

func newDNNWorkspace(batchSize int, layers []DNNLayer, dropoutRate float64) *dnnWorkspace {
	nLayers := len(layers)
	nHidden := nLayers - 1
	ws := &dnnWorkspace{
		batchSize: batchSize,
		xBatch:    make([]float64, batchSize*layers[0].InDim),
		z:         make([][]float64, nLayers),
		a:         make([][]float64, nHidden),
		prob:      make([]float64, batchSize*layers[nLayers-1].OutDim),
		dz:        make([][]float64, nLayers),
		da:        make([][]float64, nHidden),
	}
	for i := 0; i < nLayers; i++ {
		ws.z[i] = make([]float64, batchSize*layers[i].OutDim)
		ws.dz[i] = make([]float64, batchSize*layers[i].OutDim)
		if i < nHidden {
			ws.a[i] = make([]float64, batchSize*layers[i].OutDim)
			ws.da[i] = make([]float64, batchSize*layers[i].OutDim)
		}
	}
	if dropoutRate > 0 {
		ws.masks = make([][]float64, nHidden)
		for i := 0; i < nHidden; i++ {
			ws.masks[i] = make([]float64, batchSize*layers[i].OutDim)
		}
	}
	return ws
}

// workerGrads holds per-worker gradient buffers.
type workerGrads struct {
	gW [][]float64 // gW[i] for each layer
	gB [][]float64 // gB[i] for each layer
}

func newWorkerGrads(d *DNN) *workerGrads {
	wg := &workerGrads{
		gW: make([][]float64, len(d.Layers)),
		gB: make([][]float64, len(d.Layers)),
	}
	for i, layer := range d.Layers {
		wg.gW[i] = make([]float64, len(layer.W))
		wg.gB[i] = make([]float64, len(layer.B))
	}
	return wg
}

// adamState holds per-parameter momentum and variance for Adam optimizer.
type adamState struct {
	mW, vW [][]float64 // per-layer weight momentum/variance
	mB, vB [][]float64 // per-layer bias momentum/variance
	t      int         // step counter
}

func newAdamState(d *DNN) *adamState {
	s := &adamState{
		mW: make([][]float64, len(d.Layers)),
		vW: make([][]float64, len(d.Layers)),
		mB: make([][]float64, len(d.Layers)),
		vB: make([][]float64, len(d.Layers)),
	}
	for i, layer := range d.Layers {
		s.mW[i] = make([]float64, len(layer.W))
		s.vW[i] = make([]float64, len(layer.W))
		s.mB[i] = make([]float64, len(layer.B))
		s.vB[i] = make([]float64, len(layer.B))
	}
	return s
}

// TrainDNN trains the DNN on (input, target) sample pairs with mini-batch Adam.
// inputs: flat [N × InputDim], targets: [N] class indices.
// Uses parallel sub-batch processing with gradient accumulation for multi-core utilization.
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

	// Determine number of parallel workers
	workers := runtime.NumCPU()
	if workers > 8 {
		workers = 8
	}
	if workers < 1 {
		workers = 1
	}
	effectiveBatch := cfg.BatchSize * workers

	// Per-worker workspace, gradient buffers, and RNGs
	workerWSList := make([]*dnnWorkspace, workers)
	workerGradsList := make([]*workerGrads, workers)
	workerRNGs := make([]*rand.Rand, workers)
	for w := 0; w < workers; w++ {
		workerWSList[w] = newDNNWorkspace(cfg.BatchSize, dnn.Layers, dnn.DropoutRate)
		workerGradsList[w] = newWorkerGrads(dnn)
		workerRNGs[w] = rand.New(rand.NewSource(rand.Int63()))
	}

	// Total gradient accumulators
	totalGrads := newWorkerGrads(dnn)

	adam := newAdamState(dnn)

	bestValLoss := math.Inf(1)
	patience := 0

	type workerResult struct {
		loss    float64
		correct int
		samples int
	}
	results := make([]workerResult, workers)

	for epoch := 0; epoch < cfg.MaxEpochs; epoch++ {
		// Shuffle training indices
		rand.Shuffle(trainN, func(i, j int) {
			trainIdx[i], trainIdx[j] = trainIdx[j], trainIdx[i]
		})

		totalLoss := 0.0
		totalCorrect := 0
		totalSamples := 0
		nSteps := 0

		for start := 0; start < trainN; start += effectiveBatch {
			// Determine how many workers have data for this mega-batch
			activeWorkers := 0
			totalBS := 0

			var wg sync.WaitGroup
			for w := 0; w < workers; w++ {
				subStart := start + w*cfg.BatchSize
				if subStart >= trainN {
					break
				}
				subEnd := subStart + cfg.BatchSize
				if subEnd > trainN {
					subEnd = trainN
				}
				bs := subEnd - subStart
				activeWorkers++
				totalBS += bs

				wg.Add(1)
				go func(w, subStart, bs int) {
					defer wg.Done()
					ws := workerWSList[w]
					wGrads := workerGradsList[w]

					// Fill batch
					fillBatch(inputs, targets, trainIdx[subStart:subStart+bs], dnn.InputDim, ws.xBatch)
					batchTargets := make([]int, bs)
					for i := 0; i < bs; i++ {
						batchTargets[i] = targets[trainIdx[subStart+i]]
					}

					var rng *rand.Rand
					if dnn.DropoutRate > 0 {
						rng = workerRNGs[w]
					}
					loss, correct := backpropBatch(dnn, ws.xBatch, batchTargets, bs, ws, wGrads, rng)
					results[w] = workerResult{loss: loss, correct: correct, samples: bs}
				}(w, subStart, bs)
			}
			wg.Wait()

			// Accumulate gradients from all workers
			for i := range dnn.Layers {
				clearSlice(totalGrads.gW[i])
				clearSlice(totalGrads.gB[i])
			}
			for w := 0; w < activeWorkers; w++ {
				for i := range dnn.Layers {
					addSlice(totalGrads.gW[i], workerGradsList[w].gW[i])
					addSlice(totalGrads.gB[i], workerGradsList[w].gB[i])
				}
				totalLoss += results[w].loss * float64(results[w].samples)
				totalCorrect += results[w].correct
				totalSamples += results[w].samples
			}

			// Adam update with total effective batch size
			invBS := 1.0 / float64(totalBS)
			adam.t++
			for i := range dnn.Layers {
				adamUpdate(dnn.Layers[i].W, totalGrads.gW[i], adam.mW[i], adam.vW[i], cfg.LearningRate, cfg.Beta1, cfg.Beta2, cfg.Epsilon, adam.t, invBS)
				adamUpdate(dnn.Layers[i].B, totalGrads.gB[i], adam.mB[i], adam.vB[i], cfg.LearningRate, cfg.Beta1, cfg.Beta2, cfg.Epsilon, adam.t, invBS)
			}
			nSteps++
		}

		trainLoss := totalLoss / float64(totalSamples)
		trainAcc := float64(totalCorrect) / float64(totalSamples) * 100

		// Validation
		valLoss, valAcc := evaluateDNN(dnn, inputs, targets, valIdx, cfg.BatchSize, workerWSList[0])

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
// If rng is non-nil and dnn.DropoutRate > 0, dropout is applied to hidden layers.
// Returns average cross-entropy loss and number of correct predictions.
func backpropBatch(dnn *DNN, xBatch []float64, batchTargets []int, bs int,
	ws *dnnWorkspace, grads *workerGrads, rng *rand.Rand) (float64, int) {

	nLayers := len(dnn.Layers)
	O := dnn.OutputDim

	// === Forward pass ===
	prevAct := xBatch
	prevDim := dnn.InputDim

	for i := 0; i < nLayers; i++ {
		layer := &dnn.Layers[i]

		blas.Dgemm(false, true, bs, layer.OutDim, prevDim,
			1.0, prevAct, prevDim, layer.W, prevDim, 0.0, ws.z[i], layer.OutDim)

		if i < nLayers-1 {
			// Hidden layer: bias + ReLU + optional dropout
			dim := layer.OutDim
			for r := 0; r < bs; r++ {
				for j := 0; j < dim; j++ {
					idx := r*dim + j
					v := ws.z[i][idx] + layer.B[j]
					ws.z[i][idx] = v
					if v > 0 {
						ws.a[i][idx] = v
					} else {
						ws.a[i][idx] = 0
					}
				}
			}
			// Inverted dropout
			if dnn.DropoutRate > 0 && rng != nil {
				scale := 1.0 / (1.0 - dnn.DropoutRate)
				n := bs * dim
				for idx := 0; idx < n; idx++ {
					if rng.Float64() < dnn.DropoutRate {
						ws.masks[i][idx] = 0
						ws.a[i][idx] = 0
					} else {
						ws.masks[i][idx] = scale
						ws.a[i][idx] *= scale
					}
				}
			}
			prevAct = ws.a[i]
			prevDim = dim
		} else {
			// Output layer: bias + softmax + loss
			for r := 0; r < bs; r++ {
				off := r * O
				maxVal := math.Inf(-1)
				for j := 0; j < O; j++ {
					ws.z[i][off+j] += layer.B[j]
					if ws.z[i][off+j] > maxVal {
						maxVal = ws.z[i][off+j]
					}
				}
				sumExp := 0.0
				for j := 0; j < O; j++ {
					ws.prob[off+j] = math.Exp(ws.z[i][off+j] - maxVal)
					sumExp += ws.prob[off+j]
				}
				for j := 0; j < O; j++ {
					ws.prob[off+j] /= sumExp
				}
			}
		}
	}

	// Loss and accuracy
	totalLoss := 0.0
	correct := 0
	for r := 0; r < bs; r++ {
		off := r * O
		t := batchTargets[r]
		p := ws.prob[off+t]
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
			correct++
		}
	}

	// === Backward pass ===

	// dz[nLayers-1] = prob - one_hot(target)
	outIdx := nLayers - 1
	copy(ws.dz[outIdx], ws.prob[:bs*O])
	for r := 0; r < bs; r++ {
		ws.dz[outIdx][r*O+batchTargets[r]] -= 1.0
	}

	for i := nLayers - 1; i >= 0; i-- {
		layer := &dnn.Layers[i]

		// Input to this layer
		var inputToLayer []float64
		var inputDim int
		if i == 0 {
			inputToLayer = xBatch
			inputDim = dnn.InputDim
		} else {
			inputToLayer = ws.a[i-1]
			inputDim = dnn.Layers[i-1].OutDim
		}

		// gW[i] = dz[i]^T @ inputToLayer
		clearSlice(grads.gW[i])
		blas.Dgemm(true, false, layer.OutDim, inputDim, bs,
			1.0, ws.dz[i], layer.OutDim, inputToLayer, inputDim,
			0.0, grads.gW[i], inputDim)

		// gB[i] = sum(dz[i], axis=0)
		clearSlice(grads.gB[i])
		for r := 0; r < bs; r++ {
			for j := 0; j < layer.OutDim; j++ {
				grads.gB[i][j] += ws.dz[i][r*layer.OutDim+j]
			}
		}

		// Propagate gradient to previous layer
		if i > 0 {
			prevHiddenDim := dnn.Layers[i-1].OutDim

			// da[i-1] = dz[i] @ W[i]
			blas.Dgemm(false, false, bs, prevHiddenDim, layer.OutDim,
				1.0, ws.dz[i], layer.OutDim, layer.W, prevHiddenDim,
				0.0, ws.da[i-1], prevHiddenDim)

			// Apply dropout mask
			if dnn.DropoutRate > 0 && ws.masks != nil {
				n := bs * prevHiddenDim
				for idx := 0; idx < n; idx++ {
					ws.da[i-1][idx] *= ws.masks[i-1][idx]
				}
			}

			// dz[i-1] = da[i-1] * ReLU'(z[i-1])
			n := bs * prevHiddenDim
			for idx := 0; idx < n; idx++ {
				if ws.z[i-1][idx] > 0 {
					ws.dz[i-1][idx] = ws.da[i-1][idx]
				} else {
					ws.dz[i-1][idx] = 0
				}
			}
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

	nLayers := len(dnn.Layers)
	I := dnn.InputDim
	O := dnn.OutputDim

	totalLoss := 0.0
	totalCorrect := 0

	for start := 0; start < N; start += batchSize {
		end := start + batchSize
		if end > N {
			end = N
		}
		bs := end - start

		fillBatch(inputs, targets, indices[start:end], I, ws.xBatch)

		// Forward only (no dropout)
		prevAct := ws.xBatch
		prevDim := I
		for i := 0; i < nLayers; i++ {
			layer := &dnn.Layers[i]

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

		// Output layer: softmax + loss + accuracy
		outLayer := &dnn.Layers[nLayers-1]
		for r := 0; r < bs; r++ {
			off := r * O
			maxVal := math.Inf(-1)
			for j := 0; j < O; j++ {
				ws.z[nLayers-1][off+j] += outLayer.B[j]
				if ws.z[nLayers-1][off+j] > maxVal {
					maxVal = ws.z[nLayers-1][off+j]
				}
			}
			sumExp := 0.0
			for j := 0; j < O; j++ {
				e := math.Exp(ws.z[nLayers-1][off+j] - maxVal)
				ws.prob[off+j] = e
				sumExp += e
			}

			t := targets[indices[start+r]]
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

func addSlice(dst, src []float64) {
	for i := range dst {
		dst[i] += src[i]
	}
}
