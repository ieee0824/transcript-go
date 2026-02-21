package main

import (
	"bufio"
	"flag"
	"fmt"
	"math"
	"os"
	"runtime"
	"strings"
	"sync"

	"github.com/ieee0824/transcript-go/acoustic"
	"github.com/ieee0824/transcript-go/audio"
	"github.com/ieee0824/transcript-go/feature"
	"github.com/ieee0824/transcript-go/lexicon"
)

func main() {
	manifestPath := flag.String("manifest", "data/training/manifest.tsv", "path to manifest TSV (wav_path<TAB>words)")
	dictPath := flag.String("dict", "data/dict.txt", "path to pronunciation dictionary")
	amPath := flag.String("am", "data/am.gob", "path to trained GMM acoustic model (for forced alignment)")
	output := flag.String("output", "data/dnn.gob", "output DNN model path")
	hiddenDim := flag.Int("hidden", 256, "hidden layer size")
	numLayers := flag.Int("layers", 2, "number of hidden layers")
	dropout := flag.Float64("dropout", 0.0, "dropout rate for hidden layers (0=disabled)")
	contextLen := flag.Int("context", 5, "context window half-size (frames on each side)")
	lr := flag.Float64("lr", 0.001, "learning rate")
	batchSize := flag.Int("batch", 256, "mini-batch size")
	maxEpochs := flag.Int("epochs", 20, "max training epochs")
	patience := flag.Int("patience", 3, "early stopping patience (0=disabled)")
	labelSmooth := flag.Float64("label-smooth", 0.0, "label smoothing epsilon (0=disabled, recommended 0.1)")
	lrSchedule := flag.String("lr-schedule", "none", "learning rate schedule (none/cosine)")
	batchNorm := flag.Bool("batchnorm", false, "enable batch normalization on hidden layers")
	augmentFlag := flag.Bool("augment", false, "enable 5-way speed perturbation")
	manifestNoAug := flag.String("manifest-noaug", "", "additional manifest (no augmentation applied)")

	flag.Usage = func() {
		fmt.Fprintln(os.Stderr, "Usage: dnntrain [flags]")
		fmt.Fprintln(os.Stderr, "  Trains a DNN acoustic model using forced alignment labels from a GMM model.")
		fmt.Fprintln(os.Stderr)
		flag.PrintDefaults()
	}
	flag.Parse()

	workers := runtime.NumCPU()

	// Load dictionary
	dict, err := lexicon.LoadFile(*dictPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load dict: %v\n", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "Dictionary: %d words\n", len(dict.Entries))

	// Load GMM acoustic model
	amFile, err := os.Open(*amPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "open AM: %v\n", err)
		os.Exit(1)
	}
	am, err := acoustic.Load(amFile)
	amFile.Close()
	if err != nil {
		fmt.Fprintf(os.Stderr, "load AM: %v\n", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "Acoustic model: %d phonemes, %d triphones\n",
		len(am.Phonemes), len(am.Triphones))

	// Read manifest(s)
	type utterance struct {
		wavPath   string
		words     []string
		phonemes  []acoustic.Phoneme
		noAugment bool
	}

	readManifest := func(path string, noAug bool) []utterance {
		f, err := os.Open(path)
		if err != nil {
			fmt.Fprintf(os.Stderr, "open manifest: %v\n", err)
			os.Exit(1)
		}
		defer f.Close()
		var utts []utterance
		scanner := bufio.NewScanner(f)
		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if line == "" {
				continue
			}
			parts := strings.SplitN(line, "\t", 2)
			if len(parts) != 2 {
				continue
			}
			words := strings.Fields(parts[1])
			var phonemes []acoustic.Phoneme
			phonemes = append(phonemes, acoustic.PhonSil)
			allFound := true
			for _, w := range words {
				ph, ok := dict.PhonemeSequence(w)
				if !ok {
					allFound = false
					break
				}
				phonemes = append(phonemes, ph...)
			}
			if !allFound || len(phonemes) <= 1 {
				continue
			}
			phonemes = append(phonemes, acoustic.PhonSil)
			utts = append(utts, utterance{wavPath: parts[0], words: words, phonemes: phonemes, noAugment: noAug})
		}
		return utts
	}

	utts := readManifest(*manifestPath, false)
	if *manifestNoAug != "" {
		utts = append(utts, readManifest(*manifestNoAug, true)...)
	}
	fmt.Fprintf(os.Stderr, "Utterances: %d\n", len(utts))

	if len(utts) == 0 {
		fmt.Fprintln(os.Stderr, "no valid utterances found")
		os.Exit(1)
	}

	// Feature extraction (parallel, with augmentation)
	cfg := feature.DefaultConfig()
	featureDim := cfg.FeatureDim()
	factors := []float64{1.0}
	if *augmentFlag {
		factors = []float64{1.0, 0.9, 0.95, 1.05, 1.1}
	}

	type uttResult struct {
		features [][]float64
		phonemes []acoustic.Phoneme
	}
	resultCh := make(chan uttResult, len(utts)*len(factors))
	sem := make(chan struct{}, workers)
	var wg sync.WaitGroup

	for _, utt := range utts {
		wg.Add(1)
		sem <- struct{}{}
		go func(u utterance) {
			defer wg.Done()
			defer func() { <-sem }()

			samples, _, err := audio.ReadWAVFile(u.wavPath)
			if err != nil {
				return
			}

			uttFactors := factors
			if u.noAugment {
				uttFactors = []float64{1.0}
			}

			for _, factor := range uttFactors {
				augSamples := samples
				if factor != 1.0 {
					augSamples = audio.SpeedPerturb(samples, factor)
					if augSamples == nil {
						continue
					}
				}
				feats, err := feature.Extract(augSamples, cfg)
				if err != nil || len(feats) < len(u.phonemes) {
					continue
				}
				resultCh <- uttResult{features: feats, phonemes: u.phonemes}
			}
		}(utt)
	}
	go func() {
		wg.Wait()
		close(resultCh)
	}()

	var allUtts []uttResult
	for r := range resultCh {
		allUtts = append(allUtts, r)
	}
	fmt.Fprintf(os.Stderr, "Features extracted: %d utterances\n", len(allUtts))

	// Forced alignment → frame-level state labels (parallel)
	fmt.Fprintf(os.Stderr, "Running forced alignment...\n")
	type alignedUtt struct {
		features   [][]float64
		alignments []acoustic.StateAlignment
	}
	alignCh := make(chan alignedUtt, len(allUtts))
	var wgAlign sync.WaitGroup
	alignSem := make(chan struct{}, workers)

	for _, u := range allUtts {
		wgAlign.Add(1)
		alignSem <- struct{}{}
		go func(u uttResult) {
			defer wgAlign.Done()
			defer func() { <-alignSem }()

			states, err := acoustic.ForcedAlignStates(am, u.phonemes, u.features)
			if err != nil {
				return
			}
			alignCh <- alignedUtt{features: u.features, alignments: states}
		}(u)
	}
	go func() {
		wgAlign.Wait()
		close(alignCh)
	}()

	var aligned []alignedUtt
	for a := range alignCh {
		aligned = append(aligned, a)
	}
	fmt.Fprintf(os.Stderr, "Aligned: %d utterances\n", len(aligned))

	if len(aligned) == 0 {
		fmt.Fprintln(os.Stderr, "no utterances aligned")
		os.Exit(1)
	}

	// Create DNN
	dnn := acoustic.NewDNN(featureDim, *hiddenDim, *contextLen, *numLayers, *dropout, *batchNorm)

	// Count total frames and compute class priors
	totalFrames := 0
	classCounts := make([]int, dnn.OutputDim)
	for _, a := range aligned {
		for _, sa := range a.alignments {
			ci := dnn.StateClassIndex(sa.Phoneme, sa.StateIdx)
			if ci >= 0 {
				classCounts[ci]++
			}
			totalFrames++
		}
	}
	fmt.Fprintf(os.Stderr, "Total training frames: %d\n", totalFrames)

	// Compute log-prior with floor
	logPriorFloor := math.Log(1.0 / float64(totalFrames*10))
	for i, c := range classCounts {
		if c > 0 {
			dnn.LogPrior[i] = math.Log(float64(c) / float64(totalFrames))
		} else {
			dnn.LogPrior[i] = logPriorFloor
		}
	}

	// Build flat training data: inputs [N × InputDim] and targets [N]
	fmt.Fprintf(os.Stderr, "Building training data...\n")
	inputDim := dnn.InputDim
	inputs := make([]float64, totalFrames*inputDim)
	targets := make([]int, totalFrames)
	winSize := 2*(*contextLen) + 1
	idx := 0
	for _, a := range aligned {
		T := len(a.features)
		for t := 0; t < T; t++ {
			off := idx * inputDim
			for w := 0; w < winSize; w++ {
				srcT := t - *contextLen + w
				if srcT < 0 {
					srcT = 0
				} else if srcT >= T {
					srcT = T - 1
				}
				copy(inputs[off+w*featureDim:off+(w+1)*featureDim], a.features[srcT])
			}
			ci := dnn.StateClassIndex(a.alignments[t].Phoneme, a.alignments[t].StateIdx)
			if ci < 0 {
				ci = 0 // fallback for unknown phonemes
			}
			targets[idx] = ci
			idx++
		}
	}

	// Train DNN
	trainCfg := acoustic.DNNTrainConfig{
		LearningRate: *lr,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
		BatchSize:    *batchSize,
		MaxEpochs:    *maxEpochs,
		Patience:     *patience,
		HeldOutFrac:  0.1,
		LabelSmooth:  *labelSmooth,
		LRSchedule:   *lrSchedule,
	}
	totalParams := 0
	for _, l := range dnn.Layers {
		totalParams += len(l.W) + len(l.B)
	}
	fmt.Fprintf(os.Stderr, "Training DNN: input=%d hidden=%d layers=%d output=%d dropout=%.2f batchnorm=%v params=%d\n",
		dnn.InputDim, dnn.HiddenDim, len(dnn.Layers)-1, dnn.OutputDim, dnn.DropoutRate, dnn.UseBatchNorm, totalParams)
	if err := acoustic.TrainDNN(dnn, inputs, targets, trainCfg); err != nil {
		fmt.Fprintf(os.Stderr, "training error: %v\n", err)
		os.Exit(1)
	}

	// Save DNN
	outFile, err := os.Create(*output)
	if err != nil {
		fmt.Fprintf(os.Stderr, "create output: %v\n", err)
		os.Exit(1)
	}
	if err := dnn.Save(outFile); err != nil {
		outFile.Close()
		fmt.Fprintf(os.Stderr, "save DNN: %v\n", err)
		os.Exit(1)
	}
	outFile.Close()
	fmt.Fprintf(os.Stderr, "DNN saved to %s\n", *output)
}
