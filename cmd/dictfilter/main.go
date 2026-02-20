package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode"

	"github.com/ieee0824/transcript-go/acoustic"
	"github.com/ieee0824/transcript-go/lexicon"
)

func isJapanese(r rune) bool {
	return unicode.In(r, unicode.Hiragana, unicode.Katakana, unicode.Han) || r == 'ー' || r == '々'
}

func runeLen(s string) int {
	n := 0
	for range s {
		n++
	}
	return n
}

func parsePhonemes(s string) []acoustic.Phoneme {
	fields := strings.Fields(s)
	ps := make([]acoustic.Phoneme, len(fields))
	for i, f := range fields {
		ps[i] = acoustic.Phoneme(f)
	}
	return ps
}

func main() {
	corpusGlob := flag.String("corpus", "", "glob pattern for corpus files (e.g. 'training/corpus*.txt')")
	maxWords := flag.Int("max", 4000, "maximum output dictionary size")
	minEditDist := flag.Int("min-edit-dist", 0, "minimum phoneme edit distance from existing words (0=disabled, 2=recommended)")
	maxRunes := flag.Int("max-runes", 8, "maximum word length in runes")
	flag.Usage = func() {
		fmt.Fprintln(os.Stderr, "Usage: dictfilter [options] <dict.txt> <smalldict.txt>")
		fmt.Fprintln(os.Stderr, "  Filters a large dictionary to a smaller one.")
		fmt.Fprintln(os.Stderr, "  Words from smalldict.txt are always included.")
		fmt.Fprintln(os.Stderr, "  With -corpus, all words in corpus files are also included.")
		fmt.Fprintln(os.Stderr)
		flag.PrintDefaults()
	}
	flag.Parse()

	if flag.NArg() < 2 {
		flag.Usage()
		os.Exit(1)
	}

	dictPath := flag.Arg(0)
	smallDictPath := flag.Arg(1)

	// Load small dict lines (always included, as-is)
	type entry struct {
		line  string
		word  string
		phon  []acoustic.Phoneme
		nPhon int
		runeN int
	}

	var mustEntries []entry
	mustWords := make(map[string]bool)
	{
		sf, err := os.Open(smallDictPath)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		sc := bufio.NewScanner(sf)
		for sc.Scan() {
			line := strings.TrimSpace(sc.Text())
			if line == "" {
				continue
			}
			parts := strings.SplitN(line, "\t", 3)
			mustWords[parts[0]] = true
			var phon []acoustic.Phoneme
			if len(parts) >= 3 {
				phon = parsePhonemes(parts[2])
			}
			mustEntries = append(mustEntries, entry{line, parts[0], phon, len(phon), 0})
		}
		sf.Close()
	}
	fmt.Fprintf(os.Stderr, "Small dict: %d entries\n", len(mustEntries))

	// Load corpus words (if -corpus specified)
	corpusWords := make(map[string]bool)
	if *corpusGlob != "" {
		files, err := filepath.Glob(*corpusGlob)
		if err != nil {
			fmt.Fprintf(os.Stderr, "corpus glob: %v\n", err)
			os.Exit(1)
		}
		for _, path := range files {
			f, err := os.Open(path)
			if err != nil {
				fmt.Fprintf(os.Stderr, "open %s: %v\n", path, err)
				continue
			}
			sc := bufio.NewScanner(f)
			for sc.Scan() {
				for _, w := range strings.Fields(sc.Text()) {
					corpusWords[w] = true
				}
			}
			f.Close()
		}
		fmt.Fprintf(os.Stderr, "Corpus words: %d unique\n", len(corpusWords))
	}

	// Build phoneme index of must-include words for edit distance check
	var acceptedPhonemes [][]acoustic.Phoneme
	if *minEditDist > 0 {
		for _, e := range mustEntries {
			if len(e.phon) > 0 {
				acceptedPhonemes = append(acceptedPhonemes, e.phon)
			}
		}
	}

	// tooClose checks if a phoneme sequence is too close to any accepted word
	tooClose := func(phon []acoustic.Phoneme) bool {
		if *minEditDist <= 0 || len(phon) == 0 {
			return false
		}
		for _, ap := range acceptedPhonemes {
			if lexicon.PhonemeEditDistance(phon, ap) < *minEditDist {
				return true
			}
		}
		return false
	}

	// Read full dict
	f, err := os.Open(dictPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer f.Close()

	seen := make(map[string]bool)
	for w := range mustWords {
		seen[w] = true
	}

	var corpusEntries []entry
	var candidates []entry

	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		parts := strings.SplitN(line, "\t", 3)
		if len(parts) < 3 {
			continue
		}
		word := parts[0]
		if seen[word] {
			continue
		}

		phon := parsePhonemes(parts[2])
		nPhon := len(phon)
		rc := runeLen(word)

		// Corpus words: always include (no edit distance check)
		if corpusWords[word] {
			seen[word] = true
			corpusEntries = append(corpusEntries, entry{line, word, phon, nPhon, rc})
			if *minEditDist > 0 {
				acceptedPhonemes = append(acceptedPhonemes, phon)
			}
			continue
		}

		// Non-corpus candidates: apply filters
		allJP := true
		for _, r := range word {
			if !isJapanese(r) {
				allJP = false
				break
			}
		}
		if !allJP {
			continue
		}
		if rc > *maxRunes {
			continue
		}
		if nPhon < 2 {
			continue // skip single-phoneme words
		}

		seen[word] = true
		candidates = append(candidates, entry{line, word, phon, nPhon, rc})
	}

	fmt.Fprintf(os.Stderr, "Corpus words in dict: %d\n", len(corpusEntries))
	fmt.Fprintf(os.Stderr, "Other candidates: %d\n", len(candidates))

	// Sort non-corpus candidates by phoneme count, then char count
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].nPhon != candidates[j].nPhon {
			return candidates[i].nPhon < candidates[j].nPhon
		}
		return candidates[i].runeN < candidates[j].runeN
	})

	// Budget: max - small dict - corpus words
	budget := *maxWords - len(mustEntries) - len(corpusEntries)
	if budget < 0 {
		budget = 0
	}

	// Select candidates with confusion filter
	var selected []entry
	confuseSkipped := 0
	for _, c := range candidates {
		if len(selected) >= budget {
			break
		}
		if tooClose(c.phon) {
			confuseSkipped++
			continue
		}
		selected = append(selected, c)
		if *minEditDist > 0 {
			acceptedPhonemes = append(acceptedPhonemes, c.phon)
		}
	}

	// Output: small dict first, then corpus words, then selected candidates
	for _, e := range mustEntries {
		fmt.Println(e.line)
	}
	for _, e := range corpusEntries {
		fmt.Println(e.line)
	}
	for _, e := range selected {
		fmt.Println(e.line)
	}

	total := len(mustEntries) + len(corpusEntries) + len(selected)
	fmt.Fprintf(os.Stderr, "Output: %d (small=%d + corpus=%d + fill=%d)\n",
		total, len(mustEntries), len(corpusEntries), len(selected))
	if *minEditDist > 0 {
		fmt.Fprintf(os.Stderr, "Confusion filter: %d candidates rejected (min-edit-dist=%d)\n",
			confuseSkipped, *minEditDist)
	}

	// Warn about corpus words not found in dict
	missing := 0
	for w := range corpusWords {
		if !seen[w] && !mustWords[w] {
			missing++
			if missing <= 20 {
				fmt.Fprintf(os.Stderr, "WARNING: corpus word not in dict: %s\n", w)
			}
		}
	}
	if missing > 20 {
		fmt.Fprintf(os.Stderr, "WARNING: ... and %d more missing words\n", missing-20)
	}
	if missing > 0 {
		fmt.Fprintf(os.Stderr, "Total missing corpus words: %d\n", missing)
	}
}
