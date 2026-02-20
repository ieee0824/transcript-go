package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"github.com/ieee0824/transcript-go/acoustic"
	"github.com/ieee0824/transcript-go/lexicon"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "Usage: dictfix <dict.txt>")
		fmt.Fprintln(os.Stderr, "  Re-generates phoneme sequences from katakana readings using current KanaToPhonemes.")
		os.Exit(1)
	}

	f, err := os.Open(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer f.Close()

	var fixed, skipped, total int
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		parts := strings.SplitN(line, "\t", 3)
		if len(parts) < 3 {
			fmt.Println(line)
			continue
		}
		total++
		word := parts[0]
		kana := parts[1]

		phonemes := lexicon.KanaToPhonemes(kana)
		if len(phonemes) == 0 {
			skipped++
			continue
		}

		ss := make([]string, len(phonemes))
		for i, p := range phonemes {
			ss[i] = string(p)
		}
		newPhon := strings.Join(ss, " ")

		if newPhon != parts[2] {
			fixed++
		}
		fmt.Printf("%s\t%s\t%s\n", word, kana, newPhon)
	}

	// special entry
	fmt.Printf("<sil>\tSIL\t%s\n", string(acoustic.PhonSil))

	fmt.Fprintf(os.Stderr, "Total: %d, Fixed: %d, Skipped (empty phonemes): %d\n", total, fixed, skipped)
}
