package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/ieee0824/transcript-go/lexicon"
)

const mecabBatchSize = 1000

type cvEntry struct {
	clientID string
	path     string // MP3 filename in clips/
	sentence string
	upVotes  int
}

type filteredEntry struct {
	cvEntry
	words []string // MeCab tokenized, dict-filtered
}

func main() {
	cvDir := flag.String("cv-dir", "", "Common Voice directory containing validated.tsv and clips/ (required)")
	dictPath := flag.String("dict", "", "path to pronunciation dictionary (required)")
	output := flag.String("output", "", "output manifest.tsv path (required)")
	wavDir := flag.String("wav-dir", "", "output directory for converted WAV files (required)")
	minWords := flag.Int("min-words", 3, "minimum words per sentence")
	minVotes := flag.Int("min-votes", 2, "minimum up_votes for quality filter")
	workers := flag.Int("workers", runtime.NumCPU(), "number of parallel ffmpeg workers")

	flag.Usage = func() {
		fmt.Fprintln(os.Stderr, "Usage: cvimport -cv-dir DIR -dict DICT -output MANIFEST -wav-dir DIR")
		fmt.Fprintln(os.Stderr, "  Imports Mozilla Common Voice Japanese corpus for acoustic model training.")
		fmt.Fprintln(os.Stderr, "  Filters sentences by dictionary coverage, converts MP3 to 16kHz WAV.")
		fmt.Fprintln(os.Stderr)
		flag.PrintDefaults()
	}
	flag.Parse()

	if *cvDir == "" || *dictPath == "" || *output == "" || *wavDir == "" {
		fmt.Fprintln(os.Stderr, "error: -cv-dir, -dict, -output, -wav-dir are all required")
		flag.Usage()
		os.Exit(1)
	}

	// Check external tool availability
	for _, tool := range []string{"mecab", "ffmpeg"} {
		if _, err := exec.LookPath(tool); err != nil {
			fmt.Fprintf(os.Stderr, "error: %s not found in PATH\n", tool)
			os.Exit(1)
		}
	}

	// Load dictionary
	dict, err := lexicon.LoadFile(*dictPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error loading dictionary: %v\n", err)
		os.Exit(1)
	}
	wordSet := make(map[string]bool, len(dict.Entries))
	for w := range dict.Entries {
		wordSet[w] = true
	}
	fmt.Fprintf(os.Stderr, "Dictionary: %d words\n", len(wordSet))

	// Read validated.tsv
	tsvPath := filepath.Join(*cvDir, "validated.tsv")
	entries, err := readValidatedTSV(tsvPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error reading %s: %v\n", tsvPath, err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "Total entries: %d\n", len(entries))

	// Filter by up_votes
	var voteFiltered []cvEntry
	for _, e := range entries {
		if e.upVotes >= *minVotes {
			voteFiltered = append(voteFiltered, e)
		}
	}
	fmt.Fprintf(os.Stderr, "After vote filter (>=%d): %d\n", *minVotes, len(voteFiltered))

	// MeCab tokenize + dictionary filter (batch)
	filtered := filterByDict(voteFiltered, wordSet, *minWords)
	speakers := countSpeakers(filtered)
	fmt.Fprintf(os.Stderr, "After dict filter (>=%d words): %d (%.1f%%)\n",
		*minWords, len(filtered),
		safePct(len(filtered), len(voteFiltered)))
	fmt.Fprintf(os.Stderr, "Unique speakers: %d\n", speakers)

	if len(filtered) == 0 {
		fmt.Fprintln(os.Stderr, "No entries passed filters. Exiting.")
		os.Exit(0)
	}

	// Create WAV output directory
	if err := os.MkdirAll(*wavDir, 0755); err != nil {
		fmt.Fprintf(os.Stderr, "error creating wav-dir: %v\n", err)
		os.Exit(1)
	}

	// Convert MP3 → WAV (parallel)
	clipsDir := filepath.Join(*cvDir, "clips")
	var convertOK int64
	var convertFail int64
	sem := make(chan struct{}, *workers)
	var wg sync.WaitGroup

	for i := range filtered {
		wg.Add(1)
		sem <- struct{}{}
		go func(fe *filteredEntry) {
			defer wg.Done()
			defer func() { <-sem }()

			mp3Path := filepath.Join(clipsDir, fe.path)
			wavName := strings.TrimSuffix(fe.path, filepath.Ext(fe.path)) + ".wav"
			wavPath := filepath.Join(*wavDir, wavName)

			if err := convertToWAV(mp3Path, wavPath); err != nil {
				fmt.Fprintf(os.Stderr, "ffmpeg error (%s): %v\n", fe.path, err)
				atomic.AddInt64(&convertFail, 1)
				return
			}
			atomic.AddInt64(&convertOK, 1)
		}(&filtered[i])
	}
	wg.Wait()

	fmt.Fprintf(os.Stderr, "WAV converted: %d (failed: %d)\n", convertOK, convertFail)

	// Write manifest.tsv
	mf, err := os.Create(*output)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error creating output: %v\n", err)
		os.Exit(1)
	}
	w := bufio.NewWriter(mf)
	var written int
	for _, fe := range filtered {
		wavName := strings.TrimSuffix(fe.path, filepath.Ext(fe.path)) + ".wav"
		wavPath := filepath.Join(*wavDir, wavName)
		if _, err := os.Stat(wavPath); err != nil {
			continue // conversion failed
		}
		fmt.Fprintf(w, "%s\t%s\n", wavPath, strings.Join(fe.words, " "))
		written++
	}
	w.Flush()
	mf.Close()

	fmt.Fprintf(os.Stderr, "Output: %s (%d entries)\n", *output, written)
}

// readValidatedTSV reads Common Voice validated.tsv using line-by-line parsing.
// Go's csv.Reader treats " as quote characters which corrupts Japanese text fields.
func readValidatedTSV(path string) ([]cvEntry, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	// Read header
	if !scanner.Scan() {
		return nil, fmt.Errorf("empty file")
	}
	header := strings.Split(scanner.Text(), "\t")
	colIdx := make(map[string]int)
	for i, h := range header {
		colIdx[h] = i
	}
	pathCol, ok1 := colIdx["path"]
	sentCol, ok2 := colIdx["sentence"]
	upCol, ok3 := colIdx["up_votes"]
	clientCol, ok4 := colIdx["client_id"]
	if !ok1 || !ok2 || !ok3 || !ok4 {
		return nil, fmt.Errorf("missing required columns (path, sentence, up_votes, client_id)")
	}
	numCols := len(header)

	var entries []cvEntry
	for scanner.Scan() {
		fields := strings.SplitN(scanner.Text(), "\t", numCols+1)
		if len(fields) < numCols {
			continue
		}
		upVotes, _ := strconv.Atoi(fields[upCol])
		entries = append(entries, cvEntry{
			clientID: fields[clientCol],
			path:     fields[pathCol],
			sentence: fields[sentCol],
			upVotes:  upVotes,
		})
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return entries, nil
}

// filterByDict tokenizes sentences with MeCab and keeps those where all words are in the dictionary.
func filterByDict(entries []cvEntry, wordSet map[string]bool, minWords int) []filteredEntry {
	var result []filteredEntry

	for start := 0; start < len(entries); start += mecabBatchSize {
		end := start + mecabBatchSize
		if end > len(entries) {
			end = len(entries)
		}
		batch := entries[start:end]

		// Collect sentences for MeCab
		sentences := make([]string, len(batch))
		for i, e := range batch {
			// Strip punctuation that MeCab might tokenize separately
			s := strings.NewReplacer("。", "", "、", "", "？", "", "！", "", ".", "", ",", "").Replace(e.sentence)
			sentences[i] = s
		}

		tokenized, err := mecabBatch(sentences)
		if err != nil {
			fmt.Fprintf(os.Stderr, "mecab error: %v\n", err)
			continue
		}

		for i, words := range tokenized {
			if len(words) >= minWords && allInDict(words, wordSet) {
				result = append(result, filteredEntry{
					cvEntry: batch[i],
					words:   words,
				})
			}
		}
	}
	return result
}

// allInDict checks if every word exists in the dictionary.
func allInDict(words []string, wordSet map[string]bool) bool {
	for _, w := range words {
		if !wordSet[w] {
			return false
		}
	}
	return true
}

// mecabBatch tokenizes multiple lines in a single MeCab invocation.
func mecabBatch(lines []string) ([][]string, error) {
	cmd := exec.Command("mecab", "-Owakati")
	cmd.Stdin = strings.NewReader(strings.Join(lines, "\n"))
	out, err := cmd.Output()
	if err != nil {
		return nil, err
	}

	outputLines := strings.Split(strings.TrimRight(string(out), "\n"), "\n")
	result := make([][]string, len(outputLines))
	for i, line := range outputLines {
		fields := strings.Fields(strings.TrimSpace(line))
		if len(fields) > 0 {
			result[i] = fields
		}
	}
	return result, nil
}

// convertToWAV converts an MP3 file to 16kHz mono 16-bit WAV using ffmpeg.
func convertToWAV(mp3Path, wavPath string) error {
	cmd := exec.Command("ffmpeg", "-y", "-loglevel", "error",
		"-i", mp3Path,
		"-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
		"-f", "wav", wavPath)
	if output, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("%v: %s", err, string(output))
	}
	return nil
}

func countSpeakers(entries []filteredEntry) int {
	seen := make(map[string]bool)
	for _, e := range entries {
		seen[e.clientID] = true
	}
	return len(seen)
}

func safePct(n, total int) float64 {
	if total == 0 {
		return 0
	}
	return float64(n) / float64(total) * 100
}
