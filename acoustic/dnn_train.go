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
	LabelSmooth  float64 // label smoothing epsilon (0 = disabled, e.g. 0.1)
	LRSchedule   string  // "none" or "cosine"
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

	// Batch normalization intermediates (nil if !UseBatchNorm)
	bnXhat     [][]float64 // xhat[i] = normalized activations [batchSize × dim]
	bnMean     [][]float64 // batch mean [dim]
	bnInvStd   [][]float64 // 1/sqrt(var+eps) [dim]
}

func newDNNWorkspace(batchSize int, layers []DNNLayer, dropoutRate float64, useBN bool) *dnnWorkspace {
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
	if useBN {
		ws.bnXhat = make([][]float64, nHidden)
		ws.bnMean = make([][]float64, nHidden)
		ws.bnInvStd = make([][]float64, nHidden)
		for i := 0; i < nHidden; i++ {
			dim := layers[i].OutDim
			ws.bnXhat[i] = make([]float64, batchSize*dim)
			ws.bnMean[i] = make([]float64, dim)
			ws.bnInvStd[i] = make([]float64, dim)
		}
	}
	return ws
}

// workerGrads holds per-worker gradient buffers.
type workerGrads struct {
	gW [][]float64 // gW[i] for each layer
	gB [][]float64 // gB[i] for each layer

	// BN gradients (nil if !UseBatchNorm)
	gGamma [][]float64 // gGamma[i] for each hidden layer
	gBeta  [][]float64 // gBeta[i] for each hidden layer

	// BN batch statistics for running stats update
	bnMean [][]float64 // per-hidden-layer batch mean
	bnVar  [][]float64 // per-hidden-layer batch variance
	bnN    int         // batch size used for these stats
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
	if d.UseBatchNorm {
		nHidden := len(d.Layers) - 1
		wg.gGamma = make([][]float64, nHidden)
		wg.gBeta = make([][]float64, nHidden)
		wg.bnMean = make([][]float64, nHidden)
		wg.bnVar = make([][]float64, nHidden)
		for i := 0; i < nHidden; i++ {
			dim := d.BN[i].Dim
			wg.gGamma[i] = make([]float64, dim)
			wg.gBeta[i] = make([]float64, dim)
			wg.bnMean[i] = make([]float64, dim)
			wg.bnVar[i] = make([]float64, dim)
		}
	}
	return wg
}

// adamState holds per-parameter momentum and variance for Adam optimizer.
type adamState struct {
	mW, vW [][]float64 // per-layer weight momentum/variance
	mB, vB [][]float64 // per-layer bias momentum/variance
	t      int         // step counter

	// BN Adam state (nil if !UseBatchNorm)
	mGamma, vGamma [][]float64
	mBeta, vBeta   [][]float64
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
	if d.UseBatchNorm {
		nHidden := len(d.Layers) - 1
		s.mGamma = make([][]float64, nHidden)
		s.vGamma = make([][]float64, nHidden)
		s.mBeta = make([][]float64, nHidden)
		s.vBeta = make([][]float64, nHidden)
		for i := 0; i < nHidden; i++ {
			dim := d.BN[i].Dim
			s.mGamma[i] = make([]float64, dim)
			s.vGamma[i] = make([]float64, dim)
			s.mBeta[i] = make([]float64, dim)
			s.vBeta[i] = make([]float64, dim)
		}
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
		workerWSList[w] = newDNNWorkspace(cfg.BatchSize, dnn.Layers, dnn.DropoutRate, dnn.UseBatchNorm)
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
		// Compute effective learning rate
		effectiveLR := cfg.LearningRate
		if cfg.LRSchedule == "cosine" {
			lrMin := cfg.LearningRate * 0.01
			cosine := 0.5 * (1.0 + math.Cos(math.Pi*float64(epoch)/float64(cfg.MaxEpochs)))
			effectiveLR = lrMin + (cfg.LearningRate-lrMin)*cosine
		}

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
					loss, correct := backpropBatch(dnn, ws.xBatch, batchTargets, bs, ws, wGrads, rng, cfg.LabelSmooth)
					results[w] = workerResult{loss: loss, correct: correct, samples: bs}
				}(w, subStart, bs)
			}
			wg.Wait()

			// Accumulate gradients from all workers
			for i := range dnn.Layers {
				clearSlice(totalGrads.gW[i])
				clearSlice(totalGrads.gB[i])
			}
			if dnn.UseBatchNorm {
				nHidden := len(dnn.Layers) - 1
				for i := 0; i < nHidden; i++ {
					clearSlice(totalGrads.gGamma[i])
					clearSlice(totalGrads.gBeta[i])
				}
			}
			for w := 0; w < activeWorkers; w++ {
				for i := range dnn.Layers {
					addSlice(totalGrads.gW[i], workerGradsList[w].gW[i])
					addSlice(totalGrads.gB[i], workerGradsList[w].gB[i])
				}
				if dnn.UseBatchNorm {
					nHidden := len(dnn.Layers) - 1
					for i := 0; i < nHidden; i++ {
						addSlice(totalGrads.gGamma[i], workerGradsList[w].gGamma[i])
						addSlice(totalGrads.gBeta[i], workerGradsList[w].gBeta[i])
					}
				}
				totalLoss += results[w].loss * float64(results[w].samples)
				totalCorrect += results[w].correct
				totalSamples += results[w].samples
			}

			// Adam update with total effective batch size
			invBS := 1.0 / float64(totalBS)
			adam.t++
			for i := range dnn.Layers {
				adamUpdate(dnn.Layers[i].W, totalGrads.gW[i], adam.mW[i], adam.vW[i], effectiveLR, cfg.Beta1, cfg.Beta2, cfg.Epsilon, adam.t, invBS)
				adamUpdate(dnn.Layers[i].B, totalGrads.gB[i], adam.mB[i], adam.vB[i], effectiveLR, cfg.Beta1, cfg.Beta2, cfg.Epsilon, adam.t, invBS)
			}
			if dnn.UseBatchNorm {
				nHidden := len(dnn.Layers) - 1
				for i := 0; i < nHidden; i++ {
					adamUpdate(dnn.BN[i].Gamma, totalGrads.gGamma[i], adam.mGamma[i], adam.vGamma[i], effectiveLR, cfg.Beta1, cfg.Beta2, cfg.Epsilon, adam.t, invBS)
					adamUpdate(dnn.BN[i].Beta, totalGrads.gBeta[i], adam.mBeta[i], adam.vBeta[i], effectiveLR, cfg.Beta1, cfg.Beta2, cfg.Epsilon, adam.t, invBS)
				}

				// Update running stats (weighted average across workers)
				const bnMomentum = 0.1
				for i := 0; i < nHidden; i++ {
					dim := dnn.BN[i].Dim
					// Weighted mean of batch means and vars across workers
					for j := 0; j < dim; j++ {
						batchMean := 0.0
						batchVar := 0.0
						for w := 0; w < activeWorkers; w++ {
							wN := float64(results[w].samples)
							batchMean += wN * workerGradsList[w].bnMean[i][j]
							batchVar += wN * workerGradsList[w].bnVar[i][j]
						}
						batchMean /= float64(totalBS)
						batchVar /= float64(totalBS)
						// EMA update
						dnn.BN[i].RunningMean[j] = (1-bnMomentum)*dnn.BN[i].RunningMean[j] + bnMomentum*batchMean
						dnn.BN[i].RunningVar[j] = (1-bnMomentum)*dnn.BN[i].RunningVar[j] + bnMomentum*batchVar
					}
				}
			}
			nSteps++
		}

		trainLoss := totalLoss / float64(totalSamples)
		trainAcc := float64(totalCorrect) / float64(totalSamples) * 100

		// Validation
		valLoss, valAcc := evaluateDNN(dnn, inputs, targets, valIdx, cfg.BatchSize, workerWSList[0])

		if cfg.LRSchedule == "cosine" {
			fmt.Fprintf(os.Stderr, "  Epoch %2d: train_loss=%.4f train_acc=%.1f%% val_loss=%.4f val_acc=%.1f%% lr=%.6f\n",
				epoch+1, trainLoss, trainAcc, valLoss, valAcc, effectiveLR)
		} else {
			fmt.Fprintf(os.Stderr, "  Epoch %2d: train_loss=%.4f train_acc=%.1f%% val_loss=%.4f val_acc=%.1f%%\n",
				epoch+1, trainLoss, trainAcc, valLoss, valAcc)
		}

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
// labelSmooth is label smoothing epsilon (0 = disabled).
// Returns average cross-entropy loss and number of correct predictions.
func backpropBatch(dnn *DNN, xBatch []float64, batchTargets []int, bs int,
	ws *dnnWorkspace, grads *workerGrads, rng *rand.Rand, labelSmooth float64) (float64, int) {

	nLayers := len(dnn.Layers)
	O := dnn.OutputDim
	useBN := dnn.UseBatchNorm

	// === Forward pass ===
	prevAct := xBatch
	prevDim := dnn.InputDim

	for i := 0; i < nLayers; i++ {
		layer := &dnn.Layers[i]

		blas.Dgemm(false, true, bs, layer.OutDim, prevDim,
			1.0, prevAct, prevDim, layer.W, prevDim, 0.0, ws.z[i], layer.OutDim)

		if i < nLayers-1 {
			dim := layer.OutDim

			if useBN {
				// BN forward: add bias → batch normalize → gamma*xhat+beta → ReLU → dropout
				bn := &dnn.BN[i]
				bsF := float64(bs)

				// Add bias to z
				for r := 0; r < bs; r++ {
					for j := 0; j < dim; j++ {
						ws.z[i][r*dim+j] += layer.B[j]
					}
				}

				// Compute batch mean
				mean := ws.bnMean[i]
				for j := 0; j < dim; j++ {
					mean[j] = 0
				}
				for r := 0; r < bs; r++ {
					for j := 0; j < dim; j++ {
						mean[j] += ws.z[i][r*dim+j]
					}
				}
				for j := 0; j < dim; j++ {
					mean[j] /= bsF
				}

				// Compute batch variance and invStd
				invStd := ws.bnInvStd[i]
				for j := 0; j < dim; j++ {
					invStd[j] = 0
				}
				for r := 0; r < bs; r++ {
					for j := 0; j < dim; j++ {
						d := ws.z[i][r*dim+j] - mean[j]
						invStd[j] += d * d
					}
				}
				for j := 0; j < dim; j++ {
					invStd[j] = 1.0 / math.Sqrt(invStd[j]/bsF+batchNormEps)
				}

				// Normalize, apply gamma/beta, store xhat, then ReLU
				xhat := ws.bnXhat[i]
				for r := 0; r < bs; r++ {
					for j := 0; j < dim; j++ {
						idx := r*dim + j
						xh := (ws.z[i][idx] - mean[j]) * invStd[j]
						xhat[idx] = xh
						v := bn.Gamma[j]*xh + bn.Beta[j]
						// Store the BN output in z[i] for ReLU derivative tracking
						ws.z[i][idx] = v
						if v > 0 {
							ws.a[i][idx] = v
						} else {
							ws.a[i][idx] = 0
						}
					}
				}

				// Save batch stats for running stats update
				copy(grads.bnMean[i], mean)
				// Compute unbiased variance for running stats
				for j := 0; j < dim; j++ {
					// variance = 1/(invStd^2) - eps, but use the biased batch variance directly
					v := 1.0/(invStd[j]*invStd[j]) - batchNormEps
					grads.bnVar[i][j] = v
				}
				grads.bnN = bs
			} else {
				// Standard: bias + ReLU
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
	K := float64(O)
	smooth := labelSmooth / K // per-class uniform component
	for r := 0; r < bs; r++ {
		off := r * O
		t := batchTargets[r]

		if labelSmooth > 0 {
			// Smoothed cross-entropy: -Σ y_smooth[j] * log(p[j])
			targetWeight := 1.0 - labelSmooth + smooth // weight for correct class
			for j := 0; j < O; j++ {
				p := ws.prob[off+j]
				if p < 1e-30 {
					p = 1e-30
				}
				if j == t {
					totalLoss -= targetWeight * math.Log(p)
				} else {
					totalLoss -= smooth * math.Log(p)
				}
			}
		} else {
			p := ws.prob[off+t]
			if p < 1e-30 {
				p = 1e-30
			}
			totalLoss -= math.Log(p)
		}

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

	// dz[nLayers-1] = prob - y_smooth (or prob - one_hot if no smoothing)
	outIdx := nLayers - 1
	copy(ws.dz[outIdx], ws.prob[:bs*O])
	if labelSmooth > 0 {
		targetWeight := 1.0 - labelSmooth + smooth
		for r := 0; r < bs; r++ {
			off := r * O
			for j := 0; j < O; j++ {
				ws.dz[outIdx][off+j] -= smooth
			}
			ws.dz[outIdx][off+batchTargets[r]] -= (targetWeight - smooth)
		}
	} else {
		for r := 0; r < bs; r++ {
			ws.dz[outIdx][r*O+batchTargets[r]] -= 1.0
		}
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

			if useBN {
				// BN backward for hidden layer i-1
				// First apply ReLU derivative: ws.z[i-1] stores BN output (gamma*xhat+beta)
				n := bs * prevHiddenDim
				for idx := 0; idx < n; idx++ {
					if ws.z[i-1][idx] <= 0 {
						ws.da[i-1][idx] = 0
					}
				}

				// da[i-1] is now dBNout (gradient w.r.t. gamma*xhat+beta)
				bn := &dnn.BN[i-1]
				dim := prevHiddenDim
				bsF := float64(bs)
				xhat := ws.bnXhat[i-1]
				invStd := ws.bnInvStd[i-1]

				// dGamma and dBeta
				clearSlice(grads.gGamma[i-1])
				clearSlice(grads.gBeta[i-1])
				for r := 0; r < bs; r++ {
					for j := 0; j < dim; j++ {
						idx := r*dim + j
						grads.gGamma[i-1][j] += ws.da[i-1][idx] * xhat[idx]
						grads.gBeta[i-1][j] += ws.da[i-1][idx]
					}
				}

				// dxhat = da * gamma
				// dz = invStd/N * (N*dxhat - sum(dxhat) - xhat*sum(dxhat*xhat))
				sumDxhat := make([]float64, dim)
				sumDxhatXhat := make([]float64, dim)
				for r := 0; r < bs; r++ {
					for j := 0; j < dim; j++ {
						idx := r*dim + j
						dxh := ws.da[i-1][idx] * bn.Gamma[j]
						sumDxhat[j] += dxh
						sumDxhatXhat[j] += dxh * xhat[idx]
					}
				}

				for r := 0; r < bs; r++ {
					for j := 0; j < dim; j++ {
						idx := r*dim + j
						dxh := ws.da[i-1][idx] * bn.Gamma[j]
						ws.dz[i-1][idx] = invStd[j] / bsF * (bsF*dxh - sumDxhat[j] - xhat[idx]*sumDxhatXhat[j])
					}
				}
			} else {
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

		// Forward only (no dropout, BN uses running stats)
		prevAct := ws.xBatch
		prevDim := I
		for i := 0; i < nLayers; i++ {
			layer := &dnn.Layers[i]

			blas.Dgemm(false, true, bs, layer.OutDim, prevDim,
				1.0, prevAct, prevDim, layer.W, prevDim, 0.0, ws.z[i], layer.OutDim)

			if i < nLayers-1 {
				dim := layer.OutDim
				if dnn.UseBatchNorm {
					addBiasBNReLUFlat(ws.z[i], ws.a[i], layer.B, &dnn.BN[i], bs, dim)
				} else {
					for idx := 0; idx < bs*dim; idx++ {
						v := ws.z[i][idx] + layer.B[idx%dim]
						if v > 0 {
							ws.a[i][idx] = v
						} else {
							ws.a[i][idx] = 0
						}
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

// addBiasBNReLUFlat adds bias, applies BN with running stats, then ReLU.
// Reads from z, writes to a. Uses flat [rows × cols] layout.
func addBiasBNReLUFlat(z, a []float64, bias []float64, bn *BatchNormParams, rows, cols int) {
	for j := 0; j < cols; j++ {
		invStd := 1.0 / math.Sqrt(bn.RunningVar[j]+batchNormEps)
		scale := bn.Gamma[j] * invStd
		shift := bn.Beta[j] - bn.Gamma[j]*invStd*(bn.RunningMean[j]-bias[j])
		for r := 0; r < rows; r++ {
			idx := r*cols + j
			v := z[idx]*scale + shift
			if v > 0 {
				a[idx] = v
			} else {
				a[idx] = 0
			}
		}
	}
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
