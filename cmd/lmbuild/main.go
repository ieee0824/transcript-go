package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/ieee0824/transcript-go/language"
)

func main() {
	order := flag.Int("order", 2, "N-gram order (2=bigram, 3=trigram)")
	output := flag.String("output", "", "output file (default: stdout)")
	flag.Usage = func() {
		fmt.Fprintln(os.Stderr, "Usage: lmbuild [options] [input-files...]")
		fmt.Fprintln(os.Stderr, "  Builds an ARPA N-gram language model from tokenized text.")
		fmt.Fprintln(os.Stderr, "  Input: one sentence per line, words separated by spaces.")
		fmt.Fprintln(os.Stderr, "  If no input files given, reads from stdin.")
		fmt.Fprintln(os.Stderr)
		flag.PrintDefaults()
	}
	flag.Parse()

	b := language.NewBuilder(*order)

	// Read input
	var sentCount int
	if flag.NArg() == 0 {
		sentCount = readLines(b, os.Stdin)
	} else {
		for _, path := range flag.Args() {
			f, err := os.Open(path)
			if err != nil {
				fmt.Fprintf(os.Stderr, "open %s: %v\n", path, err)
				continue
			}
			sentCount += readLines(b, f)
			f.Close()
		}
	}

	// Write output
	var w *os.File
	if *output != "" {
		var err error
		w, err = os.Create(*output)
		if err != nil {
			fmt.Fprintf(os.Stderr, "create %s: %v\n", *output, err)
			os.Exit(1)
		}
		defer w.Close()
	} else {
		w = os.Stdout
	}

	if err := b.WriteARPA(w); err != nil {
		fmt.Fprintf(os.Stderr, "write ARPA: %v\n", err)
		os.Exit(1)
	}

	fmt.Fprintf(os.Stderr, "Built %d-gram model from %d sentences\n", *order, sentCount)
}

func readLines(b *language.Builder, f *os.File) int {
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
	count := 0
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		words := strings.Fields(line)
		if len(words) > 0 {
			b.AddSentence(words)
			count++
		}
	}
	return count
}
