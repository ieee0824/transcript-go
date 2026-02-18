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

func main() {
	corpusGlob := flag.String("corpus", "", "glob pattern for corpus files (e.g. 'training/corpus*.txt')")
	maxWords := flag.Int("max", 4000, "maximum output dictionary size")
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
	var mustLines []string
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
			parts := strings.SplitN(line, "\t", 2)
			mustWords[parts[0]] = true
			mustLines = append(mustLines, line)
		}
		sf.Close()
	}
	fmt.Fprintf(os.Stderr, "Small dict: %d entries\n", len(mustLines))

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

	// Read full dict and build lookup
	type entry struct {
		line  string
		word  string
		nPhon int
		runeN int
	}

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

	var corpusEntries []entry    // corpus words found in dict
	var candidates []entry       // other candidates

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

		nPhon := len(strings.Fields(parts[2]))
		rc := runeLen(word)

		// Corpus words: always include (no Japanese-only or length filter)
		if corpusWords[word] {
			seen[word] = true
			corpusEntries = append(corpusEntries, entry{line, word, nPhon, rc})
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
		if rc > 5 {
			continue
		}

		seen[word] = true
		candidates = append(candidates, entry{line, word, nPhon, rc})
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
	take := *maxWords - len(mustLines) - len(corpusEntries)
	if take < 0 {
		take = 0
	}
	if take > len(candidates) {
		take = len(candidates)
	}

	// Output: small dict first, then corpus words, then top candidates
	for _, line := range mustLines {
		fmt.Println(line)
	}
	for _, e := range corpusEntries {
		fmt.Println(e.line)
	}
	for i := 0; i < take; i++ {
		fmt.Println(candidates[i].line)
	}

	total := len(mustLines) + len(corpusEntries) + take
	fmt.Fprintf(os.Stderr, "Output: %d (small=%d + corpus=%d + fill=%d)\n",
		total, len(mustLines), len(corpusEntries), take)

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
