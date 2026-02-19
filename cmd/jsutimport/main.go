package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/ieee0824/transcript-go/lexicon"
)

const mecabBatchSize = 1000

type jsutEntry struct {
	subset   string // e.g. "basic5000"
	id       string // e.g. "BASIC5000_0001"
	sentence string
	srcWAV   string // absolute path to source WAV (48kHz)
}

type filteredEntry struct {
	jsutEntry
	words []string // MeCab tokenized, dict-filtered
}

func main() {
	jsutDir := flag.String("jsut-dir", "", "JSUT corpus root directory (required)")
	dictPath := flag.String("dict", "", "path to pronunciation dictionary (required)")
	output := flag.String("output", "", "output manifest.tsv path (required)")
	wavDir := flag.String("wav-dir", "", "output directory for resampled WAV files (required)")
	minWords := flag.Int("min-words", 3, "minimum words per sentence")
	workers := flag.Int("workers", runtime.NumCPU(), "number of parallel ffmpeg workers")

	flag.Usage = func() {
		fmt.Fprintln(os.Stderr, "Usage: jsutimport -jsut-dir DIR -dict DICT -output MANIFEST -wav-dir DIR")
		fmt.Fprintln(os.Stderr, "  Imports JSUT Japanese speech corpus for acoustic model training.")
		fmt.Fprintln(os.Stderr, "  Filters sentences by dictionary coverage, resamples WAV to 16kHz.")
		fmt.Fprintln(os.Stderr)
		flag.PrintDefaults()
	}
	flag.Parse()

	if *jsutDir == "" || *dictPath == "" || *output == "" || *wavDir == "" {
		fmt.Fprintln(os.Stderr, "error: -jsut-dir, -dict, -output, -wav-dir are all required")
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

	// Scan subsets
	entries, subsets, err := readJSUT(*jsutDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error reading JSUT: %v\n", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "Subsets found: %d (%s)\n", len(subsets), strings.Join(subsets, ", "))
	fmt.Fprintf(os.Stderr, "Total entries: %d\n", len(entries))

	// MeCab tokenize + dictionary filter (batch)
	filtered := filterByDict(entries, wordSet, *minWords)
	fmt.Fprintf(os.Stderr, "After dict filter (>=%d words): %d (%.1f%%)\n",
		*minWords, len(filtered),
		safePct(len(filtered), len(entries)))

	if len(filtered) == 0 {
		fmt.Fprintln(os.Stderr, "No entries passed filters. Exiting.")
		os.Exit(0)
	}

	// Create WAV output directory
	if err := os.MkdirAll(*wavDir, 0755); err != nil {
		fmt.Fprintf(os.Stderr, "error creating wav-dir: %v\n", err)
		os.Exit(1)
	}

	// Resample WAV 48kHz → 16kHz mono 16bit (parallel)
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

			wavPath := filepath.Join(*wavDir, fe.id+".wav")
			if err := resampleWAV(fe.srcWAV, wavPath); err != nil {
				fmt.Fprintf(os.Stderr, "ffmpeg error (%s): %v\n", fe.id, err)
				atomic.AddInt64(&convertFail, 1)
				return
			}
			atomic.AddInt64(&convertOK, 1)
		}(&filtered[i])
	}
	wg.Wait()

	fmt.Fprintf(os.Stderr, "WAV resampled: %d (failed: %d)\n", convertOK, convertFail)

	// Write manifest.tsv
	mf, err := os.Create(*output)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error creating output: %v\n", err)
		os.Exit(1)
	}
	w := bufio.NewWriter(mf)
	var written int
	for _, fe := range filtered {
		wavPath := filepath.Join(*wavDir, fe.id+".wav")
		if _, err := os.Stat(wavPath); err != nil {
			continue // resample failed
		}
		fmt.Fprintf(w, "%s\t%s\n", wavPath, strings.Join(fe.words, " "))
		written++
	}
	w.Flush()
	mf.Close()

	fmt.Fprintf(os.Stderr, "Output: %s (%d entries)\n", *output, written)
}

// readJSUT scans JSUT directory for subsets and reads all transcript files.
func readJSUT(root string) ([]jsutEntry, []string, error) {
	dirEntries, err := os.ReadDir(root)
	if err != nil {
		return nil, nil, fmt.Errorf("reading directory: %w", err)
	}

	var entries []jsutEntry
	var subsets []string

	for _, de := range dirEntries {
		if !de.IsDir() {
			continue
		}
		subsetName := de.Name()
		transcriptPath := filepath.Join(root, subsetName, "transcript_utf8.txt")
		if _, err := os.Stat(transcriptPath); err != nil {
			continue
		}

		subsetEntries, err := readTranscript(transcriptPath, subsetName, filepath.Join(root, subsetName))
		if err != nil {
			fmt.Fprintf(os.Stderr, "warning: skipping %s: %v\n", subsetName, err)
			continue
		}
		entries = append(entries, subsetEntries...)
		subsets = append(subsets, subsetName)
	}

	sort.Strings(subsets)
	return entries, subsets, nil
}

// readTranscript reads a JSUT transcript_utf8.txt file.
// Format: ID:sentence (one per line)
func readTranscript(path, subset, subsetDir string) ([]jsutEntry, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var entries []jsutEntry
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		idx := strings.IndexByte(line, ':')
		if idx < 0 {
			continue
		}
		id := line[:idx]
		sentence := line[idx+1:]
		if id == "" || sentence == "" {
			continue
		}

		srcWAV := filepath.Join(subsetDir, "wav", id+".wav")
		entries = append(entries, jsutEntry{
			subset:   subset,
			id:       id,
			sentence: sentence,
			srcWAV:   srcWAV,
		})
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return entries, nil
}

// filterByDict tokenizes sentences with MeCab and keeps those where all words are in the dictionary.
func filterByDict(entries []jsutEntry, wordSet map[string]bool, minWords int) []filteredEntry {
	var result []filteredEntry

	for start := 0; start < len(entries); start += mecabBatchSize {
		end := start + mecabBatchSize
		if end > len(entries) {
			end = len(entries)
		}
		batch := entries[start:end]

		sentences := make([]string, len(batch))
		for i, e := range batch {
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
					jsutEntry: batch[i],
					words:     words,
				})
			}
		}
	}
	return result
}

func allInDict(words []string, wordSet map[string]bool) bool {
	for _, w := range words {
		if !wordSet[w] {
			return false
		}
	}
	return true
}

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

// resampleWAV resamples a WAV file to 16kHz mono 16-bit using ffmpeg.
func resampleWAV(srcPath, dstPath string) error {
	cmd := exec.Command("ffmpeg", "-y", "-loglevel", "error",
		"-i", srcPath,
		"-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
		"-f", "wav", dstPath)
	if output, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("%v: %s", err, string(output))
	}
	return nil
}

func safePct(n, total int) float64 {
	if total == 0 {
		return 0
	}
	return float64(n) / float64(total) * 100
}
