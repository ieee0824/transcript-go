package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/ieee0824/transcript-go/acoustic"
	"github.com/ieee0824/transcript-go/audio"
	"github.com/ieee0824/transcript-go/feature"
	"github.com/ieee0824/transcript-go/lexicon"
)

type uttData struct {
	features [][]float64
	phonemes []acoustic.Phoneme
}

func main() {
	manifestPath := flag.String("manifest", "data/training/manifest.tsv", "path to manifest TSV (wav_path<TAB>words)")
	dictPath := flag.String("dict", "data/dict.txt", "path to pronunciation dictionary")
	output := flag.String("output", "data/am.gob", "output acoustic model path")
	numMix := flag.Int("mix", 1, "number of GMM components per state")
	maxIter := flag.Int("iter", 20, "max Baum-Welch iterations")
	alignIter := flag.Int("align-iter", 0, "number of forced alignment re-training iterations (0=uniform only)")
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

		// Convert words to phoneme sequence
		var phonemes []acoustic.Phoneme
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
		if !allFound || len(phonemes) == 0 {
			continue
		}

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

	var allUtts []uttData
	for i, utt := range utts {
		fmt.Fprintf(os.Stderr, "[%d/%d] %s\n", i+1, len(utts), utt.wavPath)

		samples, _, err := audio.ReadWAVFile(utt.wavPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  read WAV: %v, skipping\n", err)
			continue
		}

		features, err := feature.Extract(samples, cfg)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  extract features: %v, skipping\n", err)
			continue
		}

		T := len(features)
		N := len(utt.phonemes)
		if T < N {
			fmt.Fprintf(os.Stderr, "  too few frames (%d) for %d phonemes, skipping\n", T, N)
			continue
		}

		allUtts = append(allUtts, uttData{features: features, phonemes: utt.phonemes})
	}
	fmt.Fprintf(os.Stderr, "Valid utterances with features: %d\n", len(allUtts))

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

// segmentUniform distributes frames evenly among phonemes.
func segmentUniform(utts []uttData) map[acoustic.Phoneme][][][]float64 {
	phonemeData := make(map[acoustic.Phoneme][][][]float64)
	for _, u := range utts {
		T := len(u.features)
		N := len(u.phonemes)
		if T < N {
			continue
		}
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
