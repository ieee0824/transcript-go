package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/ieee0824/transcript-go/acoustic"
	"github.com/ieee0824/transcript-go/lexicon"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "Usage: dictconv <ipadic-csv-files...>")
		fmt.Fprintln(os.Stderr, "  Converts IPAdic CSV files to transcript dictionary format.")
		fmt.Fprintln(os.Stderr, "  Supports glob patterns: dictconv /path/to/ipadic/*.csv")
		fmt.Fprintln(os.Stderr, "  Output goes to stdout.")
		os.Exit(1)
	}

	// Expand glob patterns
	var files []string
	for _, arg := range os.Args[1:] {
		matches, err := filepath.Glob(arg)
		if err != nil {
			fmt.Fprintf(os.Stderr, "bad pattern %q: %v\n", arg, err)
			os.Exit(1)
		}
		if matches == nil {
			// No glob match — treat as literal path
			files = append(files, arg)
		} else {
			files = append(files, matches...)
		}
	}

	type entry struct {
		word     string
		reading  string
		phonemes string // space-joined phoneme string for dedup
	}

	seen := make(map[string]bool) // "word\treading\tphonemes" -> true
	var entries []entry

	for _, path := range files {
		f, err := os.Open(path)
		if err != nil {
			fmt.Fprintf(os.Stderr, "open %s: %v\n", path, err)
			continue
		}
		r := csv.NewReader(f)
		r.LazyQuotes = true
		r.FieldsPerRecord = -1 // variable fields

		for {
			record, err := r.Read()
			if err == io.EOF {
				break
			}
			if err != nil {
				continue // skip malformed lines
			}
			// IPAdic CSV: field[0]=表層形, field[11]=読み, field[12]=発音
			if len(record) < 13 {
				continue
			}
			word := record[0]
			reading := record[11]
			pronunciation := record[12]

			if pronunciation == "" || pronunciation == "*" {
				pronunciation = reading
			}
			if pronunciation == "" || pronunciation == "*" {
				continue
			}

			phonemes := lexicon.KanaToPhonemes(pronunciation)
			if len(phonemes) == 0 {
				continue
			}

			phStr := phonemeString(phonemes)
			key := word + "\t" + reading + "\t" + phStr
			if seen[key] {
				continue
			}
			seen[key] = true
			entries = append(entries, entry{word, reading, phStr})
		}
		f.Close()
	}

	// Sort by word for stable output
	sort.Slice(entries, func(i, j int) bool {
		if entries[i].word != entries[j].word {
			return entries[i].word < entries[j].word
		}
		return entries[i].reading < entries[j].reading
	})

	w := os.Stdout
	for _, e := range entries {
		fmt.Fprintf(w, "%s\t%s\t%s\n", e.word, e.reading, e.phonemes)
	}

	fmt.Fprintf(os.Stderr, "Converted %d entries from %d files\n", len(entries), len(files))
}

func phonemeString(ps []acoustic.Phoneme) string {
	ss := make([]string, len(ps))
	for i, p := range ps {
		ss[i] = string(p)
	}
	return strings.Join(ss, " ")
}
