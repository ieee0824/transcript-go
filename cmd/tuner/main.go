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
	"github.com/ieee0824/transcript-go/decoder"
	"github.com/ieee0824/transcript-go/feature"
	"github.com/ieee0824/transcript-go/language"
	"github.com/ieee0824/transcript-go/lexicon"
)

type testCase struct {
	features [][]float64
	expected string
}

type paramSet struct {
	LMWeight            float64
	WordInsertionPenalty float64
	MaxActiveTokens      int
	MaxWordEnds          int
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

	// Parse grid parameters
	lmWeights := parseFloats(*lmWeightsStr)
	wordPenalties := parseFloats(*wordPenStr)
	maxTokens := parseInts(*maxToksStr)
	maxWordEnds := parseInts(*maxWeStr)

	fmt.Fprintf(os.Stderr, "Grid: %d LMWeight × %d WordPenalty × %d MaxTokens × %d MaxWordEnds = %d combos\n",
		len(lmWeights), len(wordPenalties), len(maxTokens), len(maxWordEnds),
		len(lmWeights)*len(wordPenalties)*len(maxTokens)*len(maxWordEnds))

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

	var hiraganaSet map[string]bool
	if *hiragana {
		hiraganaSet = dict.AddHiraganaFallback()
	}

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

	// Build parameter grid
	var grid []paramSet
	for _, lw := range lmWeights {
		for _, wp := range wordPenalties {
			for _, mt := range maxTokens {
				for _, mwe := range maxWordEnds {
					grid = append(grid, paramSet{
						LMWeight:            lw,
						WordInsertionPenalty: wp,
						MaxActiveTokens:      mt,
						MaxWordEnds:          mwe,
					})
				}
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
			}
			for _, tc := range tests {
				r := decoder.Decode(tc.features, am, lm, dict, cfg)
				if r.Text == tc.expected {
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
	fmt.Printf("%-10s %-12s %-12s %-12s %8s %6s %8s\n",
		"LMWeight", "WordPenalty", "MaxTokens", "MaxWordEnds", "Correct", "Total", "Accuracy")
	fmt.Println(strings.Repeat("-", 78))
	for _, r := range results {
		acc := float64(r.correct) / float64(r.total) * 100
		fmt.Printf("%-10.1f %-12.1f %-12d %-12d %8d %6d %7.1f%%\n",
			r.params.LMWeight, r.params.WordInsertionPenalty,
			r.params.MaxActiveTokens, r.params.MaxWordEnds,
			r.correct, r.total, acc)
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
		cases = append(cases, testCase{features: feats, expected: expected})
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
