package main

import (
	"bufio"
	"flag"
	"fmt"
	"math"
	"os"
	"strings"

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
	augmentFlag := flag.Bool("augment", false, "enable 3-way speed perturbation (1.0x, 0.9x, 1.1x)")
	flag.Parse()

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
		wavPath  string
		words    []string
		phonemes []acoustic.Phoneme
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

		utts = append(utts, utterance{wavPath, words, phonemes})
	}
	fmt.Fprintf(os.Stderr, "Utterances: %d\n", len(utts))

	if len(utts) == 0 {
		fmt.Fprintln(os.Stderr, "no valid utterances found")
		os.Exit(1)
	}

	// Extract features for all utterances
	cfg := feature.DefaultConfig()
	featureDim := cfg.FeatureDim()

	// Speed perturbation factors
	factors := []float64{1.0}
	if *augmentFlag {
		factors = []float64{1.0, 0.9, 0.95, 1.05, 1.1}
	}

	var allUtts []uttData
	for i, utt := range utts {
		fmt.Fprintf(os.Stderr, "[%d/%d] %s\n", i+1, len(utts), utt.wavPath)

		samples, _, err := audio.ReadWAVFile(utt.wavPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  read WAV: %v, skipping\n", err)
			continue
		}

		// Collect per-word phoneme sequences once (independent of speed factor)
		var wordPhons [][]acoustic.Phoneme
		for _, w := range utt.words {
			ph, _ := dict.PhonemeSequence(w)
			wordPhons = append(wordPhons, ph)
		}

		for _, factor := range factors {
			augSamples := samples
			if factor != 1.0 {
				augSamples = audio.SpeedPerturb(samples, factor)
				if augSamples == nil {
					continue
				}
			}

			features, err := feature.Extract(augSamples, cfg)
			if err != nil {
				fmt.Fprintf(os.Stderr, "  extract features (%.1fx): %v, skipping\n", factor, err)
				continue
			}

			T := len(features)
			N := len(utt.phonemes)
			if T < N {
				fmt.Fprintf(os.Stderr, "  too few frames (%d) for %d phonemes (%.1fx), skipping\n", T, N, factor)
				continue
			}

			allUtts = append(allUtts, uttData{features: features, phonemes: utt.phonemes, wordPhons: wordPhons})
		}
	}
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
	trainModel(am, phonemeData, trainCfg)

	// Iterative forced alignment re-training
	for iter := 0; iter < *alignIter; iter++ {
		fmt.Fprintf(os.Stderr, "\n=== Alignment iteration %d/%d ===\n", iter+1, *alignIter)
		phonemeData = segmentAligned(am, allUtts)
		reportCoverage(phonemeData)
		trainModel(am, phonemeData, trainCfg)
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

		// Forced-align all utterances with the monophone model and collect triphone segments
		triphoneData := make(map[acoustic.Triphone][][][]float64)
		alignOK, alignFail := 0, 0

		for _, u := range allUtts {
			alignments, err := acoustic.ForcedAlign(am, u.phonemes, u.features)
			if err != nil {
				alignFail++
				continue
			}
			alignOK++

			// Map aligned phonemes to triphones.
			// Phoneme layout: [sil] + word1_phons + word2_phons + ... + [sil]
			// Alignment indices match 1:1 with u.phonemes.
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
					triphoneData[tri] = append(triphoneData[tri], u.features[a.StartFrame:a.EndFrame])
				}
				alignIdx += len(wPhons)
			}
		}

		fmt.Fprintf(os.Stderr, "  Alignment: %d ok, %d fail\n", alignOK, alignFail)
		fmt.Fprintf(os.Stderr, "  Unique triphones seen: %d\n", len(triphoneData))

		// Train triphone HMMs for triphones with sufficient data
		am.Triphones = make(map[acoustic.Triphone]*acoustic.PhonemeHMM)
		triTrainCfg := acoustic.DefaultTrainingConfig()
		triTrainCfg.MaxIterations = *maxIter

		trained, skipped := 0, 0
		for tri, segs := range triphoneData {
			if len(segs) < *minTriSeg {
				skipped++
				continue
			}
			center := tri.CenterPhoneme()
			monoHMM := am.Phonemes[center]
			if monoHMM == nil {
				skipped++
				continue
			}

			// Initialize triphone HMM from monophone
			var triHMM *acoustic.PhonemeHMM
			if triMix == *numMix {
				// Same mix count: clone monophone HMM as starting point
				triHMM = acoustic.ClonePhonemeHMM(monoHMM)
			} else {
				// Different mix count: create fresh HMM
				triHMM = acoustic.NewPhonemeHMM(center, am.FeatureDim, triMix)
			}

			if err := acoustic.TrainPhoneme(triHMM, segs, triTrainCfg); err != nil {
				fmt.Fprintf(os.Stderr, "  train %s error: %v\n", tri, err)
				skipped++
				continue
			}
			am.Triphones[tri] = triHMM
			trained++
			fmt.Fprintf(os.Stderr, "  trained %s (%d segs)\n", tri, len(segs))
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

// segmentAligned uses forced alignment with the current model.
func segmentAligned(am *acoustic.AcousticModel, utts []uttData) map[acoustic.Phoneme][][][]float64 {
	phonemeData := make(map[acoustic.Phoneme][][][]float64)
	alignOK := 0
	alignFail := 0
	for _, u := range utts {
		alignments, err := acoustic.ForcedAlign(am, u.phonemes, u.features)
		if err != nil {
			// Fall back to uniform segmentation for this utterance
			alignFail++
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
		for _, a := range alignments {
			segLen := a.EndFrame - a.StartFrame
			if segLen < 3 {
				continue
			}
			phonemeData[a.Phoneme] = append(phonemeData[a.Phoneme], u.features[a.StartFrame:a.EndFrame])
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

// trainModel trains all phoneme HMMs with the given segment data.
func trainModel(am *acoustic.AcousticModel, phonemeData map[acoustic.Phoneme][][][]float64, cfg acoustic.TrainingConfig) {
	for _, ph := range acoustic.AllPhonemes() {
		segs := phonemeData[ph]
		if len(segs) == 0 {
			fmt.Fprintf(os.Stderr, "  skipping %s (no data)\n", ph)
			continue
		}
		fmt.Fprintf(os.Stderr, "  training %s (%d segments)...\n", ph, len(segs))
		hmm := am.Phonemes[ph]
		if err := acoustic.TrainPhoneme(hmm, segs, cfg); err != nil {
			fmt.Fprintf(os.Stderr, "  train %s error: %v\n", ph, err)
		}
	}
}
