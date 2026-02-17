package main

import (
	"bufio"
	"fmt"
	"os"
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
	if len(os.Args) < 3 {
		fmt.Fprintln(os.Stderr, "usage: dictfilter <dict.txt> <smalldict.txt> [max_words=4000]")
		os.Exit(1)
	}

	maxWords := 4000
	if len(os.Args) >= 4 {
		fmt.Sscanf(os.Args[3], "%d", &maxWords)
	}

	// Load small dict lines (always included, as-is)
	var mustLines []string
	mustWords := make(map[string]bool)
	{
		sf, err := os.Open(os.Args[2])
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

	// Read and filter dict.txt
	type candidate struct {
		line   string
		nPhon  int
		runeN  int
	}
	f, err := os.Open(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer f.Close()

	seen := make(map[string]bool)
	for w := range mustWords {
		seen[w] = true
	}

	var candidates []candidate
	scanner := bufio.NewScanner(f)
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

		// Japanese chars only
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

		rc := runeLen(word)
		if rc > 5 {
			continue
		}

		nPhon := len(strings.Fields(parts[2]))

		seen[word] = true
		candidates = append(candidates, candidate{line, nPhon, rc})
	}

	fmt.Fprintf(os.Stderr, "Candidates from dict.txt: %d\n", len(candidates))

	// Sort by phoneme count (shorter = more common), then by char count
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].nPhon != candidates[j].nPhon {
			return candidates[i].nPhon < candidates[j].nPhon
		}
		return candidates[i].runeN < candidates[j].runeN
	})

	// Take top N (minus small dict entries)
	take := maxWords - len(mustLines)
	if take > len(candidates) {
		take = len(candidates)
	}

	// Output: small dict first, then top candidates
	for _, line := range mustLines {
		fmt.Println(line)
	}
	for i := 0; i < take; i++ {
		fmt.Println(candidates[i].line)
	}

	fmt.Fprintf(os.Stderr, "Output: %d (small=%d + new=%d)\n", len(mustLines)+take, len(mustLines), take)
}
