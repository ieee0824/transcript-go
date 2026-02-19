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

type uttData struct {
	features  [][]float64
	phonemes  []acoustic.Phoneme
	wordPhons [][]acoustic.Phoneme // per-word phoneme sequences (for triphone training)
}

func main() {
	manifestPath := flag.String("manifest", "data/training/manifest.tsv", "path to manifest TSV (wav_path<TAB>words)")
	dictPath := flag.String("dict", "data/dict.txt", "path to pronunciation dictionary")
	output := flag.String("output", "data/am.gob", "output acoustic model path")
	numMix := flag.Int("mix", 1, "number of GMM components per state")
	maxIter := flag.Int("iter", 20, "max Baum-Welch iterations")
	alignIter := flag.Int("align-iter", 0, "number of forced alignment re-training iterations (0=uniform only)")
	triphoneFlag := flag.Bool("triphone", false, "enable triphone training after monophone")
	minTriSeg := flag.Int("min-tri-seg", 10, "minimum segments to train a triphone HMM")
	triMixFlag := flag.Int("tri-mix", 0, "GMM components for triphone HMMs (0=min(mix,4))")
	augmentFlag := flag.Bool("augment", false, "enable 5-way speed perturbation (1.0x, 0.9x, 0.95x, 1.05x, 1.1x)")
	manifestNoAug := flag.String("manifest-noaug", "", "additional manifest (no augmentation applied)")
	flag.Parse()

	workers := runtime.NumCPU()

	// Load dictionary
	dict, err := lexicon.LoadFile(*dictPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load dict: %v\n", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "Dictionary: %d words\n", len(dict.Entries))

	// Read manifest
	mf, err := os.Open(*manifestPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "open manifest: %v\n", err)
		os.Exit(1)
	}
	defer mf.Close()

	type utterance struct {
		wavPath    string
		words      []string
		phonemes   []acoustic.Phoneme
		noAugment  bool
	}

	var utts []utterance
	scanner := bufio.NewScanner(mf)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		parts := strings.SplitN(line, "\t", 2)
		if len(parts) != 2 {
			continue
		}
		wavPath := parts[0]
		words := strings.Fields(parts[1])

		// Convert words to phoneme sequence with leading/trailing silence
		var phonemes []acoustic.Phoneme
		phonemes = append(phonemes, acoustic.PhonSil) // leading silence
		allFound := true
		for _, w := range words {
			ph, ok := dict.PhonemeSequence(w)
			if !ok {
				fmt.Fprintf(os.Stderr, "  word %q not in dict, skipping utterance\n", w)
				allFound = false
				break
			}
			phonemes = append(phonemes, ph...)
		}
		if !allFound || len(phonemes) <= 1 {
			continue
		}
		phonemes = append(phonemes, acoustic.PhonSil) // trailing silence

		utts = append(utts, utterance{wavPath: wavPath, words: words, phonemes: phonemes})
	}
	// Read additional manifest (no augmentation)
	if *manifestNoAug != "" {
		mf2, err := os.Open(*manifestNoAug)
		if err != nil {
			fmt.Fprintf(os.Stderr, "open manifest-noaug: %v\n", err)
			os.Exit(1)
		}
		scanner2 := bufio.NewScanner(mf2)
		for scanner2.Scan() {
			line := strings.TrimSpace(scanner2.Text())
			if line == "" {
				continue
			}
			parts := strings.SplitN(line, "\t", 2)
			if len(parts) != 2 {
				continue
			}
			wavPath := parts[0]
			words := strings.Fields(parts[1])

			var phonemes []acoustic.Phoneme
			phonemes = append(phonemes, acoustic.PhonSil)
			allFound := true
			for _, w := range words {
				ph, ok := dict.PhonemeSequence(w)
				if !ok {
					fmt.Fprintf(os.Stderr, "  word %q not in dict, skipping utterance\n", w)
					allFound = false
					break
				}
				phonemes = append(phonemes, ph...)
			}
			if !allFound || len(phonemes) <= 1 {
				continue
			}
			phonemes = append(phonemes, acoustic.PhonSil)

			utts = append(utts, utterance{wavPath: wavPath, words: words, phonemes: phonemes, noAugment: true})
		}
		mf2.Close()
	}

	fmt.Fprintf(os.Stderr, "Utterances: %d\n", len(utts))

	if len(utts) == 0 {
		fmt.Fprintln(os.Stderr, "no valid utterances found")
		os.Exit(1)
	}

	// Extract features for all utterances (parallel)
	cfg := feature.DefaultConfig()
	featureDim := cfg.FeatureDim()

	// Speed perturbation factors
	factors := []float64{1.0}
	if *augmentFlag {
		factors = []float64{1.0, 0.9, 0.95, 1.05, 1.1}
	}

	type indexedUttData struct {
		idx  int
		data []uttData
	}
	resultCh := make(chan indexedUttData, len(utts))
	sem := make(chan struct{}, workers)
	var wgFeat sync.WaitGroup

	for i, utt := range utts {
		wgFeat.Add(1)
		sem <- struct{}{}
		go func(idx int, utt utterance) {
			defer wgFeat.Done()
			defer func() { <-sem }()

			samples, _, err := audio.ReadWAVFile(utt.wavPath)
			if err != nil {
				fmt.Fprintf(os.Stderr, "  read WAV: %v, skipping\n", err)
				return
			}

			// Collect per-word phoneme sequences once
			var wordPhons [][]acoustic.Phoneme
			for _, w := range utt.words {
				ph, _ := dict.PhonemeSequence(w)
				wordPhons = append(wordPhons, ph)
			}

			uttFactors := factors
			if utt.noAugment {
				uttFactors = []float64{1.0}
			}

			var results []uttData
			for _, factor := range uttFactors {
				augSamples := samples
				if factor != 1.0 {
					augSamples = audio.SpeedPerturb(samples, factor)
					if augSamples == nil {
						continue
					}
				}

				features, err := feature.Extract(augSamples, cfg)
				if err != nil {
					continue
				}

				T := len(features)
				N := len(utt.phonemes)
				if T < N {
					continue
				}

				results = append(results, uttData{features: features, phonemes: utt.phonemes, wordPhons: wordPhons})
			}
			if len(results) > 0 {
				resultCh <- indexedUttData{idx: idx, data: results}
			}
		}(i, utt)
	}

	go func() {
		wgFeat.Wait()
		close(resultCh)
	}()

	var allUtts []uttData
	for r := range resultCh {
		allUtts = append(allUtts, r.data...)
	}
	fmt.Fprintf(os.Stderr, "Features extracted: %d utterances (%d workers)\n", len(allUtts), workers)

	if *augmentFlag {
		fmt.Fprintf(os.Stderr, "Valid utterances with features: %d (augmented from %d originals)\n", len(allUtts), len(utts))
	} else {
		fmt.Fprintf(os.Stderr, "Valid utterances with features: %d\n", len(allUtts))
	}

	// Initial training with uniform segmentation
	fmt.Fprintln(os.Stderr, "=== Initial training (uniform segmentation) ===")
	phonemeData := segmentUniform(allUtts)
	reportCoverage(phonemeData)

	am := acoustic.NewAcousticModel(featureDim, *numMix)
	trainCfg := acoustic.DefaultTrainingConfig()
	trainCfg.MaxIterations = *maxIter
	trainModelParallel(am, phonemeData, trainCfg, workers)

	// Iterative forced alignment re-training
	for iter := 0; iter < *alignIter; iter++ {
		fmt.Fprintf(os.Stderr, "\n=== Alignment iteration %d/%d ===\n", iter+1, *alignIter)
		phonemeData = segmentAlignedParallel(am, allUtts, workers)
		reportCoverage(phonemeData)
		trainModelParallel(am, phonemeData, trainCfg, workers)
	}

	// Triphone training phase
	if *triphoneFlag {
		fmt.Fprintln(os.Stderr, "\n=== Triphone training ===")

		triMix := *triMixFlag
		if triMix <= 0 {
			triMix = *numMix
			if triMix > 4 {
				triMix = 4
			}
		}
		fmt.Fprintf(os.Stderr, "  tri-mix=%d, min-tri-seg=%d\n", triMix, *minTriSeg)

		// Forced-align all utterances with the monophone model and collect triphone segments (parallel)
		type triAlignResult struct {
			data map[acoustic.Triphone][][][]float64
			ok   bool
		}
		triResultCh := make(chan triAlignResult, len(allUtts))
		triSem := make(chan struct{}, workers)
		var wgTriAlign sync.WaitGroup

		for _, u := range allUtts {
			wgTriAlign.Add(1)
			triSem <- struct{}{}
			go func(u uttData) {
				defer wgTriAlign.Done()
				defer func() { <-triSem }()

				alignments, err := acoustic.ForcedAlign(am, u.phonemes, u.features)
				if err != nil {
					triResultCh <- triAlignResult{ok: false}
					return
				}

				local := make(map[acoustic.Triphone][][][]float64)
				alignIdx := 1 // skip leading sil
				for _, wPhons := range u.wordPhons {
					triphones := acoustic.WordToTriphones(wPhons)
					for ti, tri := range triphones {
						ai := alignIdx + ti
						if ai >= len(alignments) {
							break
						}
						a := alignments[ai]
						segLen := a.EndFrame - a.StartFrame
						if segLen < 3 {
							continue
						}
						local[tri] = append(local[tri], u.features[a.StartFrame:a.EndFrame])
					}
					alignIdx += len(wPhons)
				}
				triResultCh <- triAlignResult{data: local, ok: true}
			}(u)
		}

		go func() {
			wgTriAlign.Wait()
			close(triResultCh)
		}()

		triphoneData := make(map[acoustic.Triphone][][][]float64)
		alignOK, alignFail := 0, 0
		for r := range triResultCh {
			if !r.ok {
				alignFail++
				continue
			}
			alignOK++
			for tri, segs := range r.data {
				triphoneData[tri] = append(triphoneData[tri], segs...)
			}
		}

		fmt.Fprintf(os.Stderr, "  Alignment: %d ok, %d fail\n", alignOK, alignFail)
		fmt.Fprintf(os.Stderr, "  Unique triphones seen: %d\n", len(triphoneData))

		// Train triphone HMMs (parallel)
		am.Triphones = make(map[acoustic.Triphone]*acoustic.PhonemeHMM)
		triTrainCfg := acoustic.DefaultTrainingConfig()
		triTrainCfg.MaxIterations = *maxIter

		type triTrainItem struct {
			tri    acoustic.Triphone
			segs   [][][]float64
			center acoustic.Phoneme
		}
		var triItems []triTrainItem
		skipped := 0
		for tri, segs := range triphoneData {
			if len(segs) < *minTriSeg {
				skipped++
				continue
			}
			center := tri.CenterPhoneme()
			if am.Phonemes[center] == nil {
				skipped++
				continue
			}
			triItems = append(triItems, triTrainItem{tri: tri, segs: segs, center: center})
		}

		type triTrainResult struct {
			tri acoustic.Triphone
			hmm *acoustic.PhonemeHMM
			err error
			n   int
		}
		triTrainCh := make(chan triTrainResult, len(triItems))
		triTrainSem := make(chan struct{}, workers)
		var wgTriTrain sync.WaitGroup

		for _, item := range triItems {
			wgTriTrain.Add(1)
			triTrainSem <- struct{}{}
			go func(item triTrainItem) {
				defer wgTriTrain.Done()
				defer func() { <-triTrainSem }()

				monoHMM := am.Phonemes[item.center]
				var triHMM *acoustic.PhonemeHMM
				if triMix == *numMix {
					triHMM = acoustic.ClonePhonemeHMM(monoHMM)
				} else {
					triHMM = acoustic.NewPhonemeHMM(item.center, am.FeatureDim, triMix)
				}

				err := acoustic.TrainPhoneme(triHMM, item.segs, triTrainCfg)
				triTrainCh <- triTrainResult{tri: item.tri, hmm: triHMM, err: err, n: len(item.segs)}
			}(item)
		}

		go func() {
			wgTriTrain.Wait()
			close(triTrainCh)
		}()

		trained := 0
		for r := range triTrainCh {
			if r.err != nil {
				fmt.Fprintf(os.Stderr, "  train %s error: %v\n", r.tri, r.err)
				skipped++
				continue
			}
			am.Triphones[r.tri] = r.hmm
			trained++
			fmt.Fprintf(os.Stderr, "  trained %s (%d segs)\n", r.tri, r.n)
		}

		fmt.Fprintf(os.Stderr, "  Triphones trained: %d, skipped: %d\n", trained, skipped)
	}

	// Save model
	of, err := os.Create(*output)
	if err != nil {
		fmt.Fprintf(os.Stderr, "create output: %v\n", err)
		os.Exit(1)
	}
	defer of.Close()

	if err := am.Save(of); err != nil {
		fmt.Fprintf(os.Stderr, "save model: %v\n", err)
		os.Exit(1)
	}

	fmt.Fprintf(os.Stderr, "Model saved to %s\n", *output)
}

// detectSilenceBounds uses c0 energy to find silence regions at utterance edges.
// Returns (speechStart, speechEnd) frame indices: features[speechStart:speechEnd] is speech.
func detectSilenceBounds(features [][]float64) (int, int) {
	T := len(features)
	if T < 6 {
		return 0, T
	}

	// c0 (first MFCC coefficient) is proportional to log energy
	minC0 := math.Inf(1)
	maxC0 := math.Inf(-1)
	for t := 0; t < T; t++ {
		c := features[t][0]
		if c < minC0 {
			minC0 = c
		}
		if c > maxC0 {
			maxC0 = c
		}
	}

	threshold := minC0 + 0.3*(maxC0-minC0)

	// Find leading silence end (search first 25% of frames)
	speechStart := 0
	limit := T / 4
	for t := 0; t < limit; t++ {
		if features[t][0] > threshold {
			break
		}
		speechStart = t + 1
	}
	if speechStart < 3 {
		speechStart = 3
	}

	// Find trailing silence start (search last 25% of frames)
	speechEnd := T
	for t := T - 1; t >= T-limit; t-- {
		if features[t][0] > threshold {
			break
		}
		speechEnd = t
	}
	if T-speechEnd < 3 {
		speechEnd = T - 3
	}

	if speechStart >= speechEnd {
		speechStart = 3
		speechEnd = T - 3
	}
	if speechEnd <= speechStart {
		return 0, T
	}
	return speechStart, speechEnd
}

// segmentUniform distributes frames evenly among phonemes.
// Silence phonemes at utterance boundaries get energy-detected silence frames.
func segmentUniform(utts []uttData) map[acoustic.Phoneme][][][]float64 {
	phonemeData := make(map[acoustic.Phoneme][][][]float64)
	for _, u := range utts {
		T := len(u.features)
		N := len(u.phonemes)
		if T < N {
			continue
		}

		// Check for leading/trailing sil phonemes
		hasLeadSil := N > 0 && u.phonemes[0] == acoustic.PhonSil
		hasTrailSil := N > 1 && u.phonemes[N-1] == acoustic.PhonSil

		if hasLeadSil || hasTrailSil {
			speechStart, speechEnd := detectSilenceBounds(u.features)

			// Assign silence segments
			if hasLeadSil && speechStart >= 3 {
				phonemeData[acoustic.PhonSil] = append(phonemeData[acoustic.PhonSil], u.features[:speechStart])
			}
			if hasTrailSil && T-speechEnd >= 3 {
				phonemeData[acoustic.PhonSil] = append(phonemeData[acoustic.PhonSil], u.features[speechEnd:])
			}

			// Distribute remaining frames among speech phonemes
			start := 0
			if hasLeadSil {
				start = 1
			}
			end := N
			if hasTrailSil {
				end = N - 1
			}
			speechPhonemes := u.phonemes[start:end]
			speechFeatures := u.features[speechStart:speechEnd]
			Ts := len(speechFeatures)
			Ns := len(speechPhonemes)
			if Ns == 0 || Ts < Ns {
				continue
			}
			framesPerPhone := Ts / Ns
			remainder := Ts % Ns
			offset := 0
			for j, ph := range speechPhonemes {
				segLen := framesPerPhone
				if j < remainder {
					segLen++
				}
				if segLen < 3 {
					continue
				}
				phonemeData[ph] = append(phonemeData[ph], speechFeatures[offset:offset+segLen])
				offset += segLen
			}
		} else {
			// No silence: uniform distribution as before
			framesPerPhone := T / N
			remainder := T % N
			offset := 0
			for j, ph := range u.phonemes {
				segLen := framesPerPhone
				if j < remainder {
					segLen++
				}
				if segLen < 3 {
					continue
				}
				phonemeData[ph] = append(phonemeData[ph], u.features[offset:offset+segLen])
				offset += segLen
			}
		}
	}
	return phonemeData
}

// segmentAlignedParallel uses forced alignment with the current model, processing utterances in parallel.
func segmentAlignedParallel(am *acoustic.AcousticModel, utts []uttData, workers int) map[acoustic.Phoneme][][][]float64 {
	type alignResult struct {
		alignments []acoustic.PhonemeAlignment
		utt        uttData
		ok         bool
	}

	resultCh := make(chan alignResult, len(utts))
	sem := make(chan struct{}, workers)
	var wg sync.WaitGroup

	for _, u := range utts {
		wg.Add(1)
		sem <- struct{}{}
		go func(u uttData) {
			defer wg.Done()
			defer func() { <-sem }()

			alignments, err := acoustic.ForcedAlign(am, u.phonemes, u.features)
			if err != nil {
				resultCh <- alignResult{utt: u, ok: false}
				return
			}
			resultCh <- alignResult{alignments: alignments, utt: u, ok: true}
		}(u)
	}

	go func() {
		wg.Wait()
		close(resultCh)
	}()

	phonemeData := make(map[acoustic.Phoneme][][][]float64)
	alignOK := 0
	alignFail := 0
	for r := range resultCh {
		if !r.ok {
			// Fall back to uniform segmentation for this utterance
			alignFail++
			u := r.utt
			T := len(u.features)
			N := len(u.phonemes)
			framesPerPhone := T / N
			remainder := T % N
			offset := 0
			for j, ph := range u.phonemes {
				segLen := framesPerPhone
				if j < remainder {
					segLen++
				}
				if segLen < 3 {
					continue
				}
				phonemeData[ph] = append(phonemeData[ph], u.features[offset:offset+segLen])
				offset += segLen
			}
			continue
		}
		alignOK++
		for _, a := range r.alignments {
			segLen := a.EndFrame - a.StartFrame
			if segLen < 3 {
				continue
			}
			phonemeData[a.Phoneme] = append(phonemeData[a.Phoneme], r.utt.features[a.StartFrame:a.EndFrame])
		}
	}
	fmt.Fprintf(os.Stderr, "  Aligned: %d, Fallback: %d\n", alignOK, alignFail)
	return phonemeData
}

// reportCoverage prints phoneme segment counts.
func reportCoverage(phonemeData map[acoustic.Phoneme][][][]float64) {
	allPhonemes := acoustic.AllPhonemes()
	covered := 0
	for _, ph := range allPhonemes {
		segs := phonemeData[ph]
		if len(segs) > 0 {
			covered++
			fmt.Fprintf(os.Stderr, "  %s: %d segments\n", ph, len(segs))
		} else {
			fmt.Fprintf(os.Stderr, "  %s: NO DATA\n", ph)
		}
	}
	fmt.Fprintf(os.Stderr, "Phoneme coverage: %d/%d\n", covered, len(allPhonemes))
}

// trainModelParallel trains all phoneme HMMs in parallel with the given segment data.
func trainModelParallel(am *acoustic.AcousticModel, phonemeData map[acoustic.Phoneme][][][]float64, cfg acoustic.TrainingConfig, workers int) {
	type trainItem struct {
		ph   acoustic.Phoneme
		segs [][][]float64
		hmm  *acoustic.PhonemeHMM
	}

	var items []trainItem
	for _, ph := range acoustic.AllPhonemes() {
		segs := phonemeData[ph]
		if len(segs) == 0 {
			fmt.Fprintf(os.Stderr, "  skipping %s (no data)\n", ph)
			continue
		}
		items = append(items, trainItem{ph: ph, segs: segs, hmm: am.Phonemes[ph]})
	}

	sem := make(chan struct{}, workers)
	var wg sync.WaitGroup

	for _, item := range items {
		wg.Add(1)
		sem <- struct{}{}
		go func(item trainItem) {
			defer wg.Done()
			defer func() { <-sem }()

			if err := acoustic.TrainPhoneme(item.hmm, item.segs, cfg); err != nil {
				fmt.Fprintf(os.Stderr, "  train %s error: %v\n", item.ph, err)
			}
			fmt.Fprintf(os.Stderr, "  trained %s (%d segments)\n", item.ph, len(item.segs))
		}(item)
	}
	wg.Wait()
}
