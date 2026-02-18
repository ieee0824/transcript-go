package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"strings"

	"github.com/ieee0824/transcript-go/lexicon"
)

const mecabBatchSize = 1000

// tagRe matches WikiExtractor <doc ...> and </doc> tags.
var tagRe = regexp.MustCompile(`^</?doc[^>]*>$`)

func main() {
	dictPath := flag.String("dict", "", "path to pronunciation dictionary (required)")
	minWords := flag.Int("min-words", 3, "minimum words per sentence")
	flag.Usage = func() {
		fmt.Fprintln(os.Stderr, "Usage: lmtext -dict DICT < input.txt > output.txt")
		fmt.Fprintln(os.Stderr, "  Reads Japanese text from stdin, tokenizes with MeCab,")
		fmt.Fprintln(os.Stderr, "  and outputs sentences where all words are in the dictionary.")
		fmt.Fprintln(os.Stderr, "  Handles WikiExtractor output (strips <doc> tags, splits on 。).")
		fmt.Fprintln(os.Stderr)
		flag.PrintDefaults()
	}
	flag.Parse()

	if *dictPath == "" {
		fmt.Fprintln(os.Stderr, "error: -dict is required")
		flag.Usage()
		os.Exit(1)
	}

	// Check MeCab availability
	if _, err := exec.LookPath("mecab"); err != nil {
		fmt.Fprintln(os.Stderr, "error: mecab not found in PATH")
		fmt.Fprintln(os.Stderr, "  install: brew install mecab mecab-ipadic")
		os.Exit(1)
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

	// Read input, preprocess, batch, and filter
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
	writer := bufio.NewWriter(os.Stdout)
	defer writer.Flush()

	var batch []string
	var totalIn, totalOut int

	flush := func() {
		if len(batch) == 0 {
			return
		}
		tokenized, err := mecabBatch(batch)
		if err != nil {
			fmt.Fprintf(os.Stderr, "mecab error: %v\n", err)
			batch = batch[:0]
			return
		}
		for _, words := range tokenized {
			if len(words) >= *minWords && allInDict(words, wordSet) {
				fmt.Fprintln(writer, strings.Join(words, " "))
				totalOut++
			}
		}
		batch = batch[:0]
	}

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || tagRe.MatchString(line) {
			continue
		}

		// Split on Japanese period for multi-sentence lines
		sentences := splitSentences(line)
		for _, sent := range sentences {
			if sent == "" {
				continue
			}
			totalIn++
			batch = append(batch, sent)
			if len(batch) >= mecabBatchSize {
				flush()
			}
		}
	}
	flush()

	if err := scanner.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "read error: %v\n", err)
	}

	rate := 0.0
	if totalIn > 0 {
		rate = float64(totalOut) / float64(totalIn) * 100
	}
	fmt.Fprintf(os.Stderr, "Input: %d sentences, Output: %d sentences (%.1f%%)\n", totalIn, totalOut, rate)
}

// splitSentences splits a line on Japanese period 。 and returns non-empty parts.
func splitSentences(line string) []string {
	parts := strings.Split(line, "。")
	result := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			result = append(result, p)
		}
	}
	return result
}

// allInDict checks if every word in the slice exists in the dictionary.
func allInDict(words []string, wordSet map[string]bool) bool {
	for _, w := range words {
		if !wordSet[w] {
			return false
		}
	}
	return true
}

// mecabBatch tokenizes multiple lines in a single MeCab invocation.
// MeCab -Owakati outputs one tokenized line per input line.
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
