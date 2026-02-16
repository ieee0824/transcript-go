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

func main() {
	manifestPath := flag.String("manifest", "data/training/manifest.tsv", "path to manifest TSV (wav_path<TAB>words)")
	dictPath := flag.String("dict", "data/dict.txt", "path to pronunciation dictionary")
	output := flag.String("output", "data/am.gob", "output acoustic model path")
	numMix := flag.Int("mix", 1, "number of GMM components per state")
	maxIter := flag.Int("iter", 20, "max Baum-Welch iterations")
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

	// Extract features and build per-phoneme training data
	cfg := feature.DefaultConfig()
	featureDim := cfg.FeatureDim()

	// phoneme -> list of segments (each segment is [][]float64)
	phonemeData := make(map[acoustic.Phoneme][][][]float64)

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

		// Uniform segmentation: distribute T frames among N phonemes
		framesPerPhone := T / N
		remainder := T % N
		offset := 0
		for j, ph := range utt.phonemes {
			segLen := framesPerPhone
			if j < remainder {
				segLen++
			}
			if segLen < 3 {
				// Need at least 3 frames for 3-state HMM
				continue
			}
			seg := features[offset : offset+segLen]
			phonemeData[ph] = append(phonemeData[ph], seg)
			offset += segLen
		}
	}

	// Report coverage
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

	// Train
	am := acoustic.NewAcousticModel(featureDim, *numMix)
	trainCfg := acoustic.DefaultTrainingConfig()
	trainCfg.MaxIterations = *maxIter

	for _, ph := range allPhonemes {
		segs := phonemeData[ph]
		if len(segs) == 0 {
			fmt.Fprintf(os.Stderr, "  skipping %s (no data)\n", ph)
			continue
		}
		fmt.Fprintf(os.Stderr, "  training %s (%d segments)...\n", ph, len(segs))
		hmm := am.Phonemes[ph]
		if err := acoustic.TrainPhoneme(hmm, segs, trainCfg); err != nil {
			fmt.Fprintf(os.Stderr, "  train %s error: %v\n", ph, err)
		}
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
