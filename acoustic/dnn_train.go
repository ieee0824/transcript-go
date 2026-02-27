package acoustic

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sync"
	"time"

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
	LRSchedule      string // "none" or "cosine"
	SpecAugFreqMask int    // max frequency mask width (0 = disabled)
	SpecAugTimeMask int    // max time mask width (0 = disabled)
	SpecAugNumFreq  int    // number of frequency masks (default 2 if FreqMask > 0)
	SpecAugNumTime  int    // number of time masks (default 1 if TimeMask > 0)
	WarmupEpochs    int    // linear LR warmup epochs (0 = disabled)
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
	xBatch    []float32   // [batchSize × InputDim]
	z         [][]float32 // z[i] = pre-activation for layer i [batchSize × layer.OutDim]
	a         [][]float32 // a[i] = post-activation for hidden layer i [batchSize × layer.OutDim]
	prob      []float64   // [batchSize × OutputDim] softmax output (float64 for loss stability)
	masks     [][]float32 // dropout masks for hidden layers (nil if no dropout)

	// Backward intermediates
	dz [][]float32 // dz[i] for each layer
	da [][]float32 // da[i] for each hidden layer

	// Batch normalization intermediates (nil if !UseBatchNorm)
	bnXhat   [][]float32 // xhat[i] = normalized activations [batchSize × dim]
	bnMean   [][]float32 // batch mean [dim]
	bnInvStd [][]float32 // 1/sqrt(var+eps) [dim]

	// Residual skip buffers (nil if !UseResidual)
	skipBuf  [2][]float32 // forward: pre-ReLU values for skip connections
	skipGrad [2][]float32 // backward: skip gradient buffers
}

func newDNNWorkspace(batchSize int, layers []DNNLayer, dropoutRate float64, useBN, useResidual bool) *dnnWorkspace {
	nLayers := len(layers)
	nHidden := nLayers - 1
	ws := &dnnWorkspace{
		batchSize: batchSize,
		xBatch:    make([]float32, batchSize*layers[0].InDim),
		z:         make([][]float32, nLayers),
		a:         make([][]float32, nHidden),
		prob:      make([]float64, batchSize*layers[nLayers-1].OutDim),
		dz:        make([][]float32, nLayers),
		da:        make([][]float32, nHidden),
	}
	for i := 0; i < nLayers; i++ {
		ws.z[i] = make([]float32, batchSize*layers[i].OutDim)
		ws.dz[i] = make([]float32, batchSize*layers[i].OutDim)
		if i < nHidden {
			ws.a[i] = make([]float32, batchSize*layers[i].OutDim)
			ws.da[i] = make([]float32, batchSize*layers[i].OutDim)
		}
	}
	if dropoutRate > 0 {
		ws.masks = make([][]float32, nHidden)
		for i := 0; i < nHidden; i++ {
			ws.masks[i] = make([]float32, batchSize*layers[i].OutDim)
		}
	}
	if useBN {
		ws.bnXhat = make([][]float32, nHidden)
		ws.bnMean = make([][]float32, nHidden)
		ws.bnInvStd = make([][]float32, nHidden)
		for i := 0; i < nHidden; i++ {
			dim := layers[i].OutDim
			ws.bnXhat[i] = make([]float32, batchSize*dim)
			ws.bnMean[i] = make([]float32, dim)
			ws.bnInvStd[i] = make([]float32, dim)
		}
	}
	if useResidual {
		n := batchSize * layers[0].OutDim
		ws.skipBuf[0] = make([]float32, n)
		ws.skipBuf[1] = make([]float32, n)
		ws.skipGrad[0] = make([]float32, n)
		ws.skipGrad[1] = make([]float32, n)
	}
	return ws
}

// workerGrads holds per-worker gradient buffers.
type workerGrads struct {
	gW [][]float32 // gW[i] for each layer
	gB [][]float32 // gB[i] for each layer

	// BN gradients (nil if !UseBatchNorm)
	gGamma [][]float32 // gGamma[i] for each hidden layer
	gBeta  [][]float32 // gBeta[i] for each hidden layer

	// BN batch statistics for running stats update
	bnMean [][]float32 // per-hidden-layer batch mean
	bnVar  [][]float32 // per-hidden-layer batch variance
	bnN    int         // batch size used for these stats
}

func newWorkerGrads(d *DNN) *workerGrads {
	wg := &workerGrads{
		gW: make([][]float32, len(d.Layers)),
		gB: make([][]float32, len(d.Layers)),
	}
	for i, layer := range d.Layers {
		wg.gW[i] = make([]float32, len(layer.W))
		wg.gB[i] = make([]float32, len(layer.B))
	}
	if d.UseBatchNorm {
		nHidden := len(d.Layers) - 1
		wg.gGamma = make([][]float32, nHidden)
		wg.gBeta = make([][]float32, nHidden)
		wg.bnMean = make([][]float32, nHidden)
		wg.bnVar = make([][]float32, nHidden)
		for i := 0; i < nHidden; i++ {
			dim := d.BN[i].Dim
			wg.gGamma[i] = make([]float32, dim)
			wg.gBeta[i] = make([]float32, dim)
			wg.bnMean[i] = make([]float32, dim)
			wg.bnVar[i] = make([]float32, dim)
		}
	}
	return wg
}

// adamState holds per-parameter momentum and variance for Adam optimizer.
type adamState struct {
	mW, vW [][]float32 // per-layer weight momentum/variance
	mB, vB [][]float32 // per-layer bias momentum/variance
	t      int         // step counter

	// BN Adam state (nil if !UseBatchNorm)
	mGamma, vGamma [][]float32
	mBeta, vBeta   [][]float32
}

func newAdamState(d *DNN) *adamState {
	s := &adamState{
		mW: make([][]float32, len(d.Layers)),
		vW: make([][]float32, len(d.Layers)),
		mB: make([][]float32, len(d.Layers)),
		vB: make([][]float32, len(d.Layers)),
	}
	for i, layer := range d.Layers {
		s.mW[i] = make([]float32, len(layer.W))
		s.vW[i] = make([]float32, len(layer.W))
		s.mB[i] = make([]float32, len(layer.B))
		s.vB[i] = make([]float32, len(layer.B))
	}
	if d.UseBatchNorm {
		nHidden := len(d.Layers) - 1
		s.mGamma = make([][]float32, nHidden)
		s.vGamma = make([][]float32, nHidden)
		s.mBeta = make([][]float32, nHidden)
		s.vBeta = make([][]float32, nHidden)
		for i := 0; i < nHidden; i++ {
			dim := d.BN[i].Dim
			s.mGamma[i] = make([]float32, dim)
			s.vGamma[i] = make([]float32, dim)
			s.mBeta[i] = make([]float32, dim)
			s.vBeta[i] = make([]float32, dim)
		}
	}
	return s
}

// TrainDNN trains the DNN on (input, target) sample pairs with mini-batch Adam.
// inputs: flat [N × InputDim] float64 (from MFCC features), targets: [N] class indices.
// Internally converts to float32 for computation.
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
		workerWSList[w] = newDNNWorkspace(cfg.BatchSize, dnn.Layers, dnn.DropoutRate, dnn.UseBatchNorm, dnn.UseResidual)
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

	// Build SpecAugment config
	var saCfg specAugmentConfig
	if cfg.SpecAugFreqMask > 0 || cfg.SpecAugTimeMask > 0 {
		numFreq := cfg.SpecAugNumFreq
		if numFreq <= 0 {
			numFreq = 2
		}
		numTime := cfg.SpecAugNumTime
		if numTime <= 0 {
			numTime = 1
		}
		saCfg = specAugmentConfig{
			FreqMaskMaxWidth: cfg.SpecAugFreqMask,
			TimeMaskMaxWidth: cfg.SpecAugTimeMask,
			NumFreqMasks:     numFreq,
			NumTimeMasks:     numTime,
			FeatureDim:       dnn.InputDim / (2*dnn.ContextLen + 1),
			ContextLen:       dnn.ContextLen,
		}
	}

	for epoch := 0; epoch < cfg.MaxEpochs; epoch++ {
		epochStart := time.Now()
		// Compute effective learning rate
		effectiveLR := cfg.LearningRate
		if cfg.WarmupEpochs > 0 && epoch < cfg.WarmupEpochs {
			// Linear warmup: ramp from ~0 to target LR
			effectiveLR = cfg.LearningRate * float64(epoch+1) / float64(cfg.WarmupEpochs)
		} else if cfg.LRSchedule == "cosine" {
			// Cosine annealing (starts after warmup if enabled)
			schedEpoch := epoch - cfg.WarmupEpochs
			schedTotal := cfg.MaxEpochs - cfg.WarmupEpochs
			if schedTotal > 0 {
				lrMin := cfg.LearningRate * 0.01
				cosine := 0.5 * (1.0 + math.Cos(math.Pi*float64(schedEpoch)/float64(schedTotal)))
				effectiveLR = lrMin + (cfg.LearningRate-lrMin)*cosine
			}
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

					// Fill batch (convert float64 inputs → float32)
					fillBatch(inputs, targets, trainIdx[subStart:subStart+bs], dnn.InputDim, ws.xBatch)
					batchTargets := make([]int, bs)
					for i := 0; i < bs; i++ {
						batchTargets[i] = targets[trainIdx[subStart+i]]
					}

					// SpecAugment on input
					if saCfg.FreqMaskMaxWidth > 0 || saCfg.TimeMaskMaxWidth > 0 {
						applySpecAugment(ws.xBatch, bs, saCfg, workerRNGs[w])
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
				clearSlice32(totalGrads.gW[i])
				clearSlice32(totalGrads.gB[i])
			}
			if dnn.UseBatchNorm {
				nHidden := len(dnn.Layers) - 1
				for i := 0; i < nHidden; i++ {
					clearSlice32(totalGrads.gGamma[i])
					clearSlice32(totalGrads.gBeta[i])
				}
			}
			for w := 0; w < activeWorkers; w++ {
				for i := range dnn.Layers {
					addSlice32(totalGrads.gW[i], workerGradsList[w].gW[i])
					addSlice32(totalGrads.gB[i], workerGradsList[w].gB[i])
				}
				if dnn.UseBatchNorm {
					nHidden := len(dnn.Layers) - 1
					for i := 0; i < nHidden; i++ {
						addSlice32(totalGrads.gGamma[i], workerGradsList[w].gGamma[i])
						addSlice32(totalGrads.gBeta[i], workerGradsList[w].gBeta[i])
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
					for j := 0; j < dim; j++ {
						batchMean := 0.0
						batchVar := 0.0
						for w := 0; w < activeWorkers; w++ {
							wN := float64(results[w].samples)
							batchMean += wN * float64(workerGradsList[w].bnMean[i][j])
							batchVar += wN * float64(workerGradsList[w].bnVar[i][j])
						}
						batchMean /= float64(totalBS)
						batchVar /= float64(totalBS)
						// EMA update
						dnn.BN[i].RunningMean[j] = float32((1-bnMomentum)*float64(dnn.BN[i].RunningMean[j]) + bnMomentum*batchMean)
						dnn.BN[i].RunningVar[j] = float32((1-bnMomentum)*float64(dnn.BN[i].RunningVar[j]) + bnMomentum*batchVar)
					}
				}
			}
			nSteps++
		}

		trainLoss := totalLoss / float64(totalSamples)
		trainAcc := float64(totalCorrect) / float64(totalSamples) * 100

		// Validation
		valLoss, valAcc := evaluateDNNParallel(dnn, inputs, targets, valIdx, cfg.BatchSize, workerWSList)

		elapsed := time.Since(epochStart)
		if cfg.LRSchedule == "cosine" || cfg.WarmupEpochs > 0 {
			fmt.Fprintf(os.Stderr, "  Epoch %2d: train_loss=%.4f train_acc=%.1f%% val_loss=%.4f val_acc=%.1f%% lr=%.6f [%s]\n",
				epoch+1, trainLoss, trainAcc, valLoss, valAcc, effectiveLR, elapsed.Round(time.Millisecond))
		} else {
			fmt.Fprintf(os.Stderr, "  Epoch %2d: train_loss=%.4f train_acc=%.1f%% val_loss=%.4f val_acc=%.1f%% [%s]\n",
				epoch+1, trainLoss, trainAcc, valLoss, valAcc, elapsed.Round(time.Millisecond))
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

// fillBatch copies float64 input features to float32 batch buffer.
func fillBatch(inputs []float64, targets []int, indices []int, inputDim int, xBatch []float32) {
	for i, idx := range indices {
		src := inputs[idx*inputDim : (idx+1)*inputDim]
		dst := xBatch[i*inputDim : (i+1)*inputDim]
		for j, v := range src {
			dst[j] = float32(v)
		}
	}
}

// specAugmentConfig holds SpecAugment masking parameters.
type specAugmentConfig struct {
	FreqMaskMaxWidth int // max frequency mask width (within FeatureDim)
	TimeMaskMaxWidth int // max time mask width (within 2*ContextLen+1)
	NumFreqMasks     int
	NumTimeMasks     int
	FeatureDim       int // per-frame feature dim (e.g. 39)
	ContextLen       int // context half-size (e.g. 5 → 11 frames total)
}

// applySpecAugment applies SpecAugment masking to xBatch in place.
func applySpecAugment(xBatch []float32, bs int, cfg specAugmentConfig, rng *rand.Rand) {
	featDim := cfg.FeatureDim
	winSize := 2*cfg.ContextLen + 1
	inputDim := winSize * featDim

	// Frequency masks: zero out band [f0, f0+w) across all context positions
	for m := 0; m < cfg.NumFreqMasks; m++ {
		w := rng.Intn(cfg.FreqMaskMaxWidth + 1)
		if w == 0 {
			continue
		}
		f0 := rng.Intn(featDim - w + 1)
		for s := 0; s < bs; s++ {
			base := s * inputDim
			for pos := 0; pos < winSize; pos++ {
				off := base + pos*featDim + f0
				for f := 0; f < w; f++ {
					xBatch[off+f] = 0
				}
			}
		}
	}

	// Time masks: zero out entire context positions [t0, t0+w)
	for m := 0; m < cfg.NumTimeMasks; m++ {
		w := rng.Intn(cfg.TimeMaskMaxWidth + 1)
		if w == 0 {
			continue
		}
		t0 := rng.Intn(winSize - w + 1)
		for s := 0; s < bs; s++ {
			base := s * inputDim
			for pos := t0; pos < t0+w; pos++ {
				off := base + pos*featDim
				for f := 0; f < featDim; f++ {
					xBatch[off+f] = 0
				}
			}
		}
	}
}

// backpropBatch computes forward pass, loss, and gradients for one mini-batch.
func backpropBatch(dnn *DNN, xBatch []float32, batchTargets []int, bs int,
	ws *dnnWorkspace, grads *workerGrads, rng *rand.Rand, labelSmooth float64) (float64, int) {

	nLayers := len(dnn.Layers)
	O := dnn.OutputDim
	useBN := dnn.UseBatchNorm
	useRes := dnn.UseResidual

	// Clear skip buffers for this batch
	if useRes {
		for k := range ws.skipBuf[0] {
			ws.skipBuf[0][k] = 0
		}
		for k := range ws.skipBuf[1] {
			ws.skipBuf[1][k] = 0
		}
	}

	// === Forward pass ===
	prevAct := xBatch
	prevDim := dnn.InputDim

	for i := 0; i < nLayers; i++ {
		layer := &dnn.Layers[i]

		blas.Sgemm(false, true, bs, layer.OutDim, prevDim,
			1.0, prevAct, prevDim, layer.W, prevDim, 0.0, ws.z[i], layer.OutDim)

		if i < nLayers-1 {
			dim := layer.OutDim

			if useBN {
				bn := &dnn.BN[i]
				bsF := float32(bs)

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
					invStd[j] = float32(1.0 / math.Sqrt(float64(invStd[j]/bsF+batchNormEps)))
				}

				// Normalize, apply gamma/beta, store xhat
				xhat := ws.bnXhat[i]
				for r := 0; r < bs; r++ {
					for j := 0; j < dim; j++ {
						idx := r*dim + j
						xh := (ws.z[i][idx] - mean[j]) * invStd[j]
						xhat[idx] = xh
						ws.z[i][idx] = bn.Gamma[j]*xh + bn.Beta[j]
					}
				}

				// Residual: add skip from layer i-2, save pre-ReLU
				if useRes {
					n := bs * dim
					if i >= 2 {
						for k := 0; k < n; k++ {
							ws.z[i][k] += ws.skipBuf[i%2][k]
						}
					}
					copy(ws.skipBuf[i%2], ws.z[i][:n])
				}

				// ReLU
				for idx := 0; idx < bs*dim; idx++ {
					if ws.z[i][idx] > 0 {
						ws.a[i][idx] = ws.z[i][idx]
					} else {
						ws.a[i][idx] = 0
					}
				}

				// Save batch stats for running stats update
				copy(grads.bnMean[i], mean)
				for j := 0; j < dim; j++ {
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
				scale := float32(1.0 / (1.0 - dnn.DropoutRate))
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
			// Output layer: bias + softmax (in float64 for loss stability)
			for r := 0; r < bs; r++ {
				off := r * O
				// Add bias
				for j := 0; j < O; j++ {
					ws.z[i][off+j] += layer.B[j]
				}
				// Softmax in float64
				maxVal := float64(ws.z[i][off])
				for j := 1; j < O; j++ {
					if v := float64(ws.z[i][off+j]); v > maxVal {
						maxVal = v
					}
				}
				sumExp := 0.0
				for j := 0; j < O; j++ {
					ws.prob[off+j] = math.Exp(float64(ws.z[i][off+j]) - maxVal)
					sumExp += ws.prob[off+j]
				}
				for j := 0; j < O; j++ {
					ws.prob[off+j] /= sumExp
				}
			}
		}
	}

	// Loss and accuracy (float64)
	totalLoss := 0.0
	correct := 0
	K := float64(O)
	smooth := labelSmooth / K
	for r := 0; r < bs; r++ {
		off := r * O
		t := batchTargets[r]

		if labelSmooth > 0 {
			targetWeight := 1.0 - labelSmooth + smooth
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

	// Clear residual skip gradient buffers
	if useRes {
		for k := range ws.skipGrad[0] {
			ws.skipGrad[0][k] = 0
		}
		for k := range ws.skipGrad[1] {
			ws.skipGrad[1][k] = 0
		}
	}

	// dz[nLayers-1] = prob - y_smooth (convert float64 prob → float32 dz)
	outIdx := nLayers - 1
	if labelSmooth > 0 {
		targetWeight := 1.0 - labelSmooth + smooth
		for r := 0; r < bs; r++ {
			off := r * O
			for j := 0; j < O; j++ {
				ws.dz[outIdx][off+j] = float32(ws.prob[off+j] - smooth)
			}
			ws.dz[outIdx][off+batchTargets[r]] = float32(ws.prob[off+batchTargets[r]] - targetWeight)
		}
	} else {
		for r := 0; r < bs; r++ {
			off := r * O
			for j := 0; j < O; j++ {
				ws.dz[outIdx][off+j] = float32(ws.prob[off+j])
			}
			ws.dz[outIdx][off+batchTargets[r]] -= 1.0
		}
	}

	for i := nLayers - 1; i >= 0; i-- {
		layer := &dnn.Layers[i]

		// Input to this layer
		var inputToLayer []float32
		var inputDim int
		if i == 0 {
			inputToLayer = xBatch
			inputDim = dnn.InputDim
		} else {
			inputToLayer = ws.a[i-1]
			inputDim = dnn.Layers[i-1].OutDim
		}

		// gW[i] = dz[i]^T @ inputToLayer
		clearSlice32(grads.gW[i])
		blas.Sgemm(true, false, layer.OutDim, inputDim, bs,
			1.0, ws.dz[i], layer.OutDim, inputToLayer, inputDim,
			0.0, grads.gW[i], inputDim)

		// gB[i] = sum(dz[i], axis=0)
		clearSlice32(grads.gB[i])
		for r := 0; r < bs; r++ {
			for j := 0; j < layer.OutDim; j++ {
				grads.gB[i][j] += ws.dz[i][r*layer.OutDim+j]
			}
		}

		// Propagate gradient to previous layer
		if i > 0 {
			prevHiddenDim := dnn.Layers[i-1].OutDim

			// da[i-1] = dz[i] @ W[i]
			blas.Sgemm(false, false, bs, prevHiddenDim, layer.OutDim,
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
				n := bs * prevHiddenDim
				for idx := 0; idx < n; idx++ {
					if ws.z[i-1][idx] <= 0 {
						ws.da[i-1][idx] = 0
					}
				}

				// Residual gradient
				if useRes {
					h := i - 1
					nHidden := nLayers - 1
					if h+2 < nHidden {
						for k := 0; k < n; k++ {
							ws.da[h][k] += ws.skipGrad[h%2][k]
						}
					}
					if h >= 2 {
						copy(ws.skipGrad[h%2], ws.da[h][:n])
					}
				}

				bn := &dnn.BN[i-1]
				dim := prevHiddenDim
				bsF := float32(bs)
				xhat := ws.bnXhat[i-1]
				invStd := ws.bnInvStd[i-1]

				// dGamma and dBeta
				clearSlice32(grads.gGamma[i-1])
				clearSlice32(grads.gBeta[i-1])
				for r := 0; r < bs; r++ {
					for j := 0; j < dim; j++ {
						idx := r*dim + j
						grads.gGamma[i-1][j] += ws.da[i-1][idx] * xhat[idx]
						grads.gBeta[i-1][j] += ws.da[i-1][idx]
					}
				}

				// dxhat = da * gamma
				// dz = invStd/N * (N*dxhat - sum(dxhat) - xhat*sum(dxhat*xhat))
				sumDxhat := make([]float32, dim)
				sumDxhatXhat := make([]float32, dim)
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

// adamUpdate applies one Adam step on float32 parameters.
// Hyperparameters (lr, beta, eps) are float64; computation uses float64 internally.
func adamUpdate(params, grad, m, v []float32, lr, beta1, beta2, eps float64, t int, gradScale float64) {
	bc1 := 1.0 - math.Pow(beta1, float64(t))
	bc2 := 1.0 - math.Pow(beta2, float64(t))
	for i := range params {
		g := float64(grad[i]) * gradScale
		mi := beta1*float64(m[i]) + (1-beta1)*g
		vi := beta2*float64(v[i]) + (1-beta2)*g*g
		m[i] = float32(mi)
		v[i] = float32(vi)
		mHat := mi / bc1
		vHat := vi / bc2
		params[i] -= float32(lr * mHat / (math.Sqrt(vHat) + eps))
	}
}

// evaluateDNNParallel computes average loss and accuracy using multiple workers.
func evaluateDNNParallel(dnn *DNN, inputs []float64, targets []int, indices []int, batchSize int, wsList []*dnnWorkspace) (float64, float64) {
	N := len(indices)
	if N == 0 {
		return 0, 0
	}
	workers := len(wsList)
	if workers < 1 {
		workers = 1
	}

	// Split indices evenly across workers
	chunkSize := (N + workers - 1) / workers

	type evalResult struct {
		loss    float64
		correct int
		count   int
	}
	results := make([]evalResult, workers)

	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		s := w * chunkSize
		if s >= N {
			break
		}
		e := s + chunkSize
		if e > N {
			e = N
		}
		wg.Add(1)
		go func(w, s, e int) {
			defer wg.Done()
			loss, correct := evaluateDNNChunk(dnn, inputs, targets, indices[s:e], batchSize, wsList[w])
			results[w] = evalResult{loss: loss, correct: correct, count: e - s}
		}(w, s, e)
	}
	wg.Wait()

	totalLoss := 0.0
	totalCorrect := 0
	totalN := 0
	for _, r := range results {
		totalLoss += r.loss
		totalCorrect += r.correct
		totalN += r.count
	}
	if totalN == 0 {
		return 0, 0
	}
	return totalLoss / float64(totalN), float64(totalCorrect) / float64(totalN) * 100
}

// evaluateDNNChunk computes raw loss sum and correct count on a chunk of data.
func evaluateDNNChunk(dnn *DNN, inputs []float64, targets []int, indices []int, batchSize int, ws *dnnWorkspace) (float64, int) {
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

			blas.Sgemm(false, true, bs, layer.OutDim, prevDim,
				1.0, prevAct, prevDim, layer.W, prevDim, 0.0, ws.z[i], layer.OutDim)

			if i < nLayers-1 {
				dim := layer.OutDim
				if dnn.UseBatchNorm {
					if dnn.UseResidual {
						addBiasBN(ws.z[i], layer.B, &dnn.BN[i], bs, dim)
						n := bs * dim
						if i >= 2 {
							for k := 0; k < n; k++ {
								ws.z[i][k] += ws.skipBuf[i%2][k]
							}
						}
						copy(ws.skipBuf[i%2], ws.z[i][:n])
						applyReLU(ws.z[i], n)
						copy(ws.a[i][:n], ws.z[i][:n])
					} else {
						addBiasBNReLU(ws.z[i], layer.B, &dnn.BN[i], bs, dim)
						copy(ws.a[i][:bs*dim], ws.z[i][:bs*dim])
					}
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

		// Output layer: softmax + loss + accuracy (float64 for stability)
		outLayer := &dnn.Layers[nLayers-1]
		for r := 0; r < bs; r++ {
			off := r * O
			maxVal := float64(ws.z[nLayers-1][off])
			for j := 1; j < O; j++ {
				ws.z[nLayers-1][off+j] += outLayer.B[j]
				if v := float64(ws.z[nLayers-1][off+j]); v > maxVal {
					maxVal = v
				}
			}
			// Add bias to first element (loop above starts at j=1)
			ws.z[nLayers-1][off] += outLayer.B[0]
			if v := float64(ws.z[nLayers-1][off]); v > maxVal {
				maxVal = v
			}

			sumExp := 0.0
			probs := make([]float64, O)
			for j := 0; j < O; j++ {
				probs[j] = math.Exp(float64(ws.z[nLayers-1][off+j]) - maxVal)
				sumExp += probs[j]
			}

			t := targets[indices[start+r]]
			p := probs[t] / sumExp
			if p < 1e-30 {
				p = 1e-30
			}
			totalLoss -= math.Log(p)

			bestJ := 0
			bestP := probs[0]
			for j := 1; j < O; j++ {
				if probs[j] > bestP {
					bestP = probs[j]
					bestJ = j
				}
			}
			if bestJ == t {
				totalCorrect++
			}
		}
	}

	return totalLoss, totalCorrect
}

func clearSlice32(s []float32) {
	for i := range s {
		s[i] = 0
	}
}

func addSlice32(dst, src []float32) {
	for i := range dst {
		dst[i] += src[i]
	}
}
