package main

import (
	"bufio"
	"flag"
	"fmt"
	"math"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/ieee0824/transcript-go/acoustic"
	"github.com/ieee0824/transcript-go/audio"
	"github.com/ieee0824/transcript-go/correct"
	"github.com/ieee0824/transcript-go/decoder"
	"github.com/ieee0824/transcript-go/feature"
	"github.com/ieee0824/transcript-go/language"
	"github.com/ieee0824/transcript-go/lexicon"
)

type testCase struct {
	features [][]float64
	expected string
	path     string
}

type paramSet struct {
	LMWeight            float64
	WordInsertionPenalty float64
	MaxActiveTokens      int
	MaxWordEnds          int
	RescoreWeight        float64
	WPPerFrame           float64
	ShortBeamThreshold   int
}

type result struct {
	params  paramSet
	correct int
	total   int
}

func main() {
	amPath := flag.String("am", "", "path to acoustic model")
	dnnPath := flag.String("dnn", "", "path to DNN model")
	lmPath := flag.String("lm", "", "path to LM (ARPA)")
	dictPath := flag.String("dict", "", "path to dictionary")
	manifests := flag.String("manifest", "", "comma-separated manifest.tsv paths")
	beam := flag.Float64("beam", 200.0, "beam width (fixed)")
	lmWeightsStr := flag.String("lm-weights", "8,10,12,15,20", "comma-separated LM weights")
	wordPenStr := flag.String("word-penalties", "-5,-2,-1,0,1", "comma-separated word penalties")
	maxToksStr := flag.String("max-tokens", "1000,2000,3000", "comma-separated max active tokens")
	maxWeStr := flag.String("max-word-ends", "0,30,50,100", "comma-separated max word ends")
	workers := flag.Int("workers", 0, "parallel workers (default: NumCPU)")
	shard := flag.Int("shard", 0, "shard index for distributed execution (0-based)")
	numShards := flag.Int("num-shards", 1, "total number of shards (1 = no sharding)")
	oovProb := flag.Float64("oov-prob", 0, "OOV log10 probability")
	lmInterp := flag.Float64("lm-interp", 0.0, "LM interpolation weight")
	hiragana := flag.Bool("hiragana", false, "enable hiragana fallback")
	hiraganaPenalty := flag.Float64("hiragana-penalty", -15.0, "penalty for hiragana fallback words")
	verbose := flag.Bool("verbose", false, "show per-utterance errors for best params")
	nbest := flag.Int("nbest", 0, "N-best count for rescoring (0=disabled)")
	rescoreLMPath := flag.String("rescore-lm", "", "path to rescoring LM (ARPA)")
	rescoreWeightsStr := flag.String("rescore-weights", "1,2,3,5", "comma-separated rescore LM weights")
	wpPerFrameStr := flag.String("wp-per-frame", "0", "comma-separated WP per-frame adjustments (effectiveWP = WP + wpPerFrame*T)")
	shortBeamStr := flag.String("short-beam-thresholds", "0", "comma-separated short beam thresholds in frames (0=default 80)")
	correctDictPath := flag.String("correct-dict", "", "large dictionary for post-correction")
	correctLMPath := flag.String("correct-lm", "", "large LM for post-correction")
	correctMaxDist := flag.Int("correct-max-dist", 2, "max phoneme edit distance for correction candidates")
	correctKeepBonus := flag.Float64("correct-keep-bonus", 0, "bonus for keeping original word (higher = more conservative)")

	flag.Usage = func() {
		fmt.Fprintln(os.Stderr, "Usage: tuner -am AM -lm LM -dict DICT -manifest M1,M2,...")
		fmt.Fprintln(os.Stderr, "  Grid search decoder parameters against test manifests.")
		fmt.Fprintln(os.Stderr)
		flag.PrintDefaults()
	}
	flag.Parse()

	if *amPath == "" || *lmPath == "" || *dictPath == "" || *manifests == "" {
		flag.Usage()
		os.Exit(1)
	}

	if *workers <= 0 {
		*workers = runtime.NumCPU()
	}

	// Load models
	fmt.Fprintln(os.Stderr, "Loading models...")
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

	if *dnnPath != "" {
		f, err := os.Open(*dnnPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "open DNN: %v\n", err)
			os.Exit(1)
		}
		dnn, err := acoustic.LoadDNN(f)
		f.Close()
		if err != nil {
			fmt.Fprintf(os.Stderr, "load DNN: %v\n", err)
			os.Exit(1)
		}
		am.DNN = dnn
	}

	lmFile, err := os.Open(*lmPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "open LM: %v\n", err)
		os.Exit(1)
	}
	lm, err := language.LoadARPA(lmFile)
	lmFile.Close()
	if err != nil {
		fmt.Fprintf(os.Stderr, "load LM: %v\n", err)
		os.Exit(1)
	}
	if *oovProb != 0 {
		lm.OOVLogProb = *oovProb * math.Ln10
	}

	dict, err := lexicon.LoadFile(*dictPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load dict: %v\n", err)
		os.Exit(1)
	}

	// Load rescoring LM if specified
	var rescoreLM *language.NGramModel
	if *rescoreLMPath != "" && *nbest > 1 {
		rlmFile, err := os.Open(*rescoreLMPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "open rescore LM: %v\n", err)
			os.Exit(1)
		}
		rescoreLM, err = language.LoadARPA(rlmFile)
		rlmFile.Close()
		if err != nil {
			fmt.Fprintf(os.Stderr, "load rescore LM: %v\n", err)
			os.Exit(1)
		}
		fmt.Fprintln(os.Stderr, "Loaded rescoring LM")
	}

	var hiraganaSet map[string]bool
	if *hiragana {
		hiraganaSet = dict.AddHiraganaFallback()
	}

	// Load post-correction model if specified
	var corrector *correct.Corrector
	if *correctDictPath != "" && *correctLMPath != "" {
		fmt.Fprintln(os.Stderr, "Loading correction models...")
		cDictFile, err := os.Open(*correctDictPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "open correct dict: %v\n", err)
			os.Exit(1)
		}
		cDict, err := lexicon.Load(cDictFile)
		cDictFile.Close()
		if err != nil {
			fmt.Fprintf(os.Stderr, "load correct dict: %v\n", err)
			os.Exit(1)
		}
		cLMFile, err := os.Open(*correctLMPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "open correct LM: %v\n", err)
			os.Exit(1)
		}
		cLM, err := language.LoadARPA(cLMFile)
		cLMFile.Close()
		if err != nil {
			fmt.Fprintf(os.Stderr, "load correct LM: %v\n", err)
			os.Exit(1)
		}
		fmt.Fprintf(os.Stderr, "Building confusion model (large dict: %d words)...\n", len(cDict.Entries))
		confusion := correct.BuildConfusionModel(cDict, dict)
		corrector = &correct.Corrector{
			LM:              cLM,
			Confusion:       confusion,
			LMWeight:        1.0,
			ConfusionWeight: 1.0,
			BeamWidth:       50,
			MaxDist:         *correctMaxDist,
			KeepBonus:       *correctKeepBonus,
		}
		fmt.Fprintln(os.Stderr, "Correction model ready")
	}

	// Parse grid parameters
	lmWeights := parseFloats(*lmWeightsStr)
	wordPenalties := parseFloats(*wordPenStr)
	maxTokens := parseInts(*maxToksStr)
	maxWordEnds := parseInts(*maxWeStr)
	rescoreWeights := parseFloats(*rescoreWeightsStr)
	wpPerFrames := parseFloats(*wpPerFrameStr)
	shortBeamThresholds := parseInts(*shortBeamStr)

	// If rescoring is not enabled, use a single dummy weight so grid loop works
	if rescoreLM == nil {
		rescoreWeights = []float64{0}
	}

	combos := len(lmWeights) * len(wordPenalties) * len(maxTokens) * len(maxWordEnds) * len(rescoreWeights) * len(wpPerFrames) * len(shortBeamThresholds)
	fmt.Fprintf(os.Stderr, "Grid: %d LMWeight × %d WordPenalty × %d MaxTokens × %d MaxWordEnds × %d WPPerFrame × %d ShortBeam",
		len(lmWeights), len(wordPenalties), len(maxTokens), len(maxWordEnds), len(wpPerFrames), len(shortBeamThresholds))
	if rescoreLM != nil {
		fmt.Fprintf(os.Stderr, " × %d RescoreWeight", len(rescoreWeights))
	}
	fmt.Fprintf(os.Stderr, " = %d combos\n", combos)

	// Load and pre-extract features from all manifests
	fmt.Fprintln(os.Stderr, "Extracting features...")
	var tests []testCase
	for _, mpath := range strings.Split(*manifests, ",") {
		mpath = strings.TrimSpace(mpath)
		if mpath == "" {
			continue
		}
		loaded := loadManifest(mpath)
		tests = append(tests, loaded...)
	}
	fmt.Fprintf(os.Stderr, "Loaded %d test files\n", len(tests))

	// Build parameter grid (base dimensions, then expand with WPPerFrame)
	var grid []paramSet
	for _, lw := range lmWeights {
		for _, wp := range wordPenalties {
			for _, mt := range maxTokens {
				for _, mwe := range maxWordEnds {
					for _, rw := range rescoreWeights {
						grid = append(grid, paramSet{
							LMWeight:            lw,
							WordInsertionPenalty: wp,
							MaxActiveTokens:      mt,
							MaxWordEnds:          mwe,
							RescoreWeight:        rw,
						})
					}
				}
			}
		}
	}
	if len(wpPerFrames) > 1 || (len(wpPerFrames) == 1 && wpPerFrames[0] != 0) {
		base := grid
		grid = make([]paramSet, 0, len(base)*len(wpPerFrames))
		for _, ps := range base {
			for _, wppf := range wpPerFrames {
				ps.WPPerFrame = wppf
				grid = append(grid, ps)
			}
		}
	}
	if len(shortBeamThresholds) > 1 || (len(shortBeamThresholds) == 1 && shortBeamThresholds[0] != 0) {
		base := grid
		grid = make([]paramSet, 0, len(base)*len(shortBeamThresholds))
		for _, ps := range base {
			for _, sbt := range shortBeamThresholds {
				ps.ShortBeamThreshold = sbt
				grid = append(grid, ps)
			}
		}
	}

	// Shard filtering
	if *numShards > 1 {
		if *shard < 0 || *shard >= *numShards {
			fmt.Fprintf(os.Stderr, "shard must be in [0, %d)\n", *numShards)
			os.Exit(1)
		}
		var sharded []paramSet
		for i, ps := range grid {
			if i%*numShards == *shard {
				sharded = append(sharded, ps)
			}
		}
		fmt.Fprintf(os.Stderr, "Shard %d/%d: %d combinations (of %d total)\n", *shard, *numShards, len(sharded), len(grid))
		grid = sharded
	}

	// Run grid search in parallel
	fmt.Fprintf(os.Stderr, "Running %d combinations on %d workers...\n", len(grid), *workers)
	results := make([]result, len(grid))
	var wg sync.WaitGroup
	sem := make(chan struct{}, *workers)
	var done int64
	total := int64(len(grid))
	startTime := time.Now()

	// Progress reporter
	stopProgress := make(chan struct{})
	go func() {
		ticker := time.NewTicker(5 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-stopProgress:
				return
			case <-ticker.C:
				d := atomic.LoadInt64(&done)
				if d == 0 {
					fmt.Fprintf(os.Stderr, "  [%s] %d/%d (0%%) ...\n",
						time.Since(startTime).Truncate(time.Second), d, total)
					continue
				}
				elapsed := time.Since(startTime)
				perCombo := elapsed / time.Duration(d)
				remaining := perCombo * time.Duration(total-d)
				fmt.Fprintf(os.Stderr, "  [%s] %d/%d (%.0f%%) ETA %s\n",
					elapsed.Truncate(time.Second), d, total,
					float64(d)/float64(total)*100,
					remaining.Truncate(time.Second))
			}
		}
	}()

	for gi, ps := range grid {
		wg.Add(1)
		sem <- struct{}{}
		go func(gi int, ps paramSet) {
			defer wg.Done()
			defer func() { <-sem }()
			correct := 0
			cfg := decoder.Config{
				BeamWidth:            *beam,
				MaxActiveTokens:      ps.MaxActiveTokens,
				LMWeight:             ps.LMWeight,
				WordInsertionPenalty: ps.WordInsertionPenalty,
				LMInterpolation:     *lmInterp,
				MaxWordEnds:          ps.MaxWordEnds,
				HiraganaSet:          hiraganaSet,
				HiraganaPenalty:      *hiraganaPenalty,
				NBestCount:           *nbest,
				WPPerFrame:           ps.WPPerFrame,
				ShortBeamThreshold:   ps.ShortBeamThreshold,
			}
			for _, tc := range tests {
				r := decoder.Decode(tc.features, am, lm, dict, cfg)
				hyp := r.Text
				if rescoreLM != nil && len(r.NBest) > 0 {
					rescored := decoder.RescoreNBest(r.NBest, rescoreLM, ps.RescoreWeight)
					hyp = rescored.Text
				}
				if corrector != nil {
					hyp = corrector.CorrectText(hyp)
				}
				if hyp == tc.expected {
					correct++
				}
			}
			results[gi] = result{params: ps, correct: correct, total: len(tests)}
			atomic.AddInt64(&done, 1)
		}(gi, ps)
	}
	wg.Wait()
	close(stopProgress)
	fmt.Fprintf(os.Stderr, "Completed %d combinations in %s\n", total, time.Since(startTime).Truncate(time.Second))

	// Sort by accuracy descending, then by LMWeight ascending for ties
	sort.Slice(results, func(i, j int) bool {
		if results[i].correct != results[j].correct {
			return results[i].correct > results[j].correct
		}
		return results[i].params.LMWeight < results[j].params.LMWeight
	})

	// Print results
	showWPPF := len(wpPerFrames) > 1 || (len(wpPerFrames) == 1 && wpPerFrames[0] != 0)
	showSBT := len(shortBeamThresholds) > 1 || (len(shortBeamThresholds) == 1 && shortBeamThresholds[0] != 0)
	if rescoreLM != nil {
		fmt.Printf("%-10s %-12s %-12s %-12s %-12s", "LMWeight", "WordPenalty", "MaxTokens", "MaxWordEnds", "RescoreWt")
	} else {
		fmt.Printf("%-10s %-12s %-12s %-12s", "LMWeight", "WordPenalty", "MaxTokens", "MaxWordEnds")
	}
	if showWPPF {
		fmt.Printf(" %-10s", "WPPerFrm")
	}
	if showSBT {
		fmt.Printf(" %-8s", "SBThresh")
	}
	fmt.Printf(" %8s %6s %8s\n", "Correct", "Total", "Accuracy")
	fmt.Println(strings.Repeat("-", 100))
	for _, r := range results {
		acc := float64(r.correct) / float64(r.total) * 100
		if rescoreLM != nil {
			fmt.Printf("%-10.1f %-12.1f %-12d %-12d %-12.1f",
				r.params.LMWeight, r.params.WordInsertionPenalty,
				r.params.MaxActiveTokens, r.params.MaxWordEnds,
				r.params.RescoreWeight)
		} else {
			fmt.Printf("%-10.1f %-12.1f %-12d %-12d",
				r.params.LMWeight, r.params.WordInsertionPenalty,
				r.params.MaxActiveTokens, r.params.MaxWordEnds)
		}
		if showWPPF {
			fmt.Printf(" %-10.4f", r.params.WPPerFrame)
		}
		if showSBT {
			fmt.Printf(" %-8d", r.params.ShortBeamThreshold)
		}
		fmt.Printf(" %8d %6d %7.1f%%\n", r.correct, r.total, acc)
	}

	// Verbose: re-run best params and show per-utterance errors
	if *verbose && len(results) > 0 {
		best := results[0].params
		hdr := fmt.Sprintf("LW=%.1f WP=%.1f MT=%d MWE=%d",
			best.LMWeight, best.WordInsertionPenalty, best.MaxActiveTokens, best.MaxWordEnds)
		if best.WPPerFrame != 0 {
			hdr += fmt.Sprintf(" WPPF=%.4f", best.WPPerFrame)
		}
		if best.ShortBeamThreshold != 0 {
			hdr += fmt.Sprintf(" SBT=%d", best.ShortBeamThreshold)
		}
		if rescoreLM != nil {
			hdr += fmt.Sprintf(" RW=%.1f", best.RescoreWeight)
		}
		fmt.Fprintf(os.Stderr, "\n=== Verbose: errors with best params (%s) ===\n", hdr)
		cfg := decoder.Config{
			BeamWidth:            *beam,
			MaxActiveTokens:      best.MaxActiveTokens,
			LMWeight:             best.LMWeight,
			WordInsertionPenalty: best.WordInsertionPenalty,
			LMInterpolation:     *lmInterp,
			MaxWordEnds:          best.MaxWordEnds,
			HiraganaSet:          hiraganaSet,
			HiraganaPenalty:      *hiraganaPenalty,
			NBestCount:           *nbest,
			WPPerFrame:           best.WPPerFrame,
			ShortBeamThreshold:   best.ShortBeamThreshold,
		}
		errCount := 0
		for _, tc := range tests {
			r := decoder.Decode(tc.features, am, lm, dict, cfg)
			hyp := r.Text
			raw := r.Text
			if rescoreLM != nil && len(r.NBest) > 0 {
				rescored := decoder.RescoreNBest(r.NBest, rescoreLM, best.RescoreWeight)
				hyp = rescored.Text
			}
			if corrector != nil {
				hyp = corrector.CorrectText(hyp)
			}
			if hyp != tc.expected {
				errCount++
				if corrector != nil && hyp != raw {
					fmt.Fprintf(os.Stderr, "  [MISS] expected: %-30s got: %-30s (raw: %-30s) file: %s\n",
						tc.expected, hyp, raw, tc.path)
				} else if rescoreLM != nil && len(r.NBest) > 0 {
					fmt.Fprintf(os.Stderr, "  [MISS] expected: %-30s got: %-30s (1best: %-30s) file: %s\n",
						tc.expected, hyp, r.Text, tc.path)
					for ni, nb := range r.NBest {
						tag := "  "
						if nb.Text == tc.expected {
							tag = ">>"
						}
						fmt.Fprintf(os.Stderr, "    %s [%2d] score=%.2f text: %s\n", tag, ni+1, nb.LogScore, nb.Text)
					}
				} else {
					fmt.Fprintf(os.Stderr, "  [MISS] expected: %-30s got: %-30s file: %s\n", tc.expected, hyp, tc.path)
				}
			} else if corrector != nil && hyp != raw {
				fmt.Fprintf(os.Stderr, "  [FIX!] expected: %-30s raw: %-30s corrected: %s\n",
					tc.expected, raw, hyp)
			}
		}
		fmt.Fprintf(os.Stderr, "Total errors: %d/%d\n", errCount, len(tests))
	}
}

func loadManifest(path string) []testCase {
	f, err := os.Open(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "open manifest %s: %v\n", path, err)
		return nil
	}
	defer f.Close()

	featCfg := feature.DefaultConfig()
	var cases []testCase
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.SplitN(line, "\t", 2)
		if len(parts) != 2 {
			continue
		}
		wavPath := parts[0]
		expected := parts[1]

		samples, _, err := audio.ReadWAVFile(wavPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "read %s: %v\n", wavPath, err)
			continue
		}
		feats, err := feature.Extract(samples, featCfg)
		if err != nil {
			fmt.Fprintf(os.Stderr, "features %s: %v\n", wavPath, err)
			continue
		}
		cases = append(cases, testCase{features: feats, expected: expected, path: wavPath})
	}
	return cases
}

func parseFloats(s string) []float64 {
	var vals []float64
	for _, part := range strings.Split(s, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		v, err := strconv.ParseFloat(part, 64)
		if err != nil {
			fmt.Fprintf(os.Stderr, "invalid float %q: %v\n", part, err)
			continue
		}
		vals = append(vals, v)
	}
	return vals
}

func parseInts(s string) []int {
	var vals []int
	for _, part := range strings.Split(s, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		v, err := strconv.Atoi(part)
		if err != nil {
			fmt.Fprintf(os.Stderr, "invalid int %q: %v\n", part, err)
			continue
		}
		vals = append(vals, v)
	}
	return vals
}
