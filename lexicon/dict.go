package lexicon

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strings"
	"github.com/ieee0824/transcript-go/acoustic"
)

// Entry represents a single pronunciation for a word.
type Entry struct {
	Word     string
	Reading  string             // kana reading
	Phonemes []acoustic.Phoneme // phoneme sequence
}

// Dictionary holds word-to-pronunciation mappings.
type Dictionary struct {
	Entries map[string][]Entry // word -> list of alternative pronunciations
}

// NewDictionary creates an empty dictionary.
func NewDictionary() *Dictionary {
	return &Dictionary{
		Entries: make(map[string][]Entry),
	}
}

// Add adds a pronunciation entry to the dictionary.
func (d *Dictionary) Add(word, reading string, phonemes []acoustic.Phoneme) {
	d.Entries[word] = append(d.Entries[word], Entry{
		Word:     word,
		Reading:  reading,
		Phonemes: phonemes,
	})
}

// Load reads a pronunciation dictionary from a tab-separated file.
// Format: word<TAB>reading<TAB>phoneme1 phoneme2 phoneme3 ...
func Load(r io.Reader) (*Dictionary, error) {
	d := NewDictionary()
	scanner := bufio.NewScanner(r)
	lineNum := 0

	for scanner.Scan() {
		lineNum++
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		parts := strings.SplitN(line, "\t", 3)
		if len(parts) < 3 {
			return nil, fmt.Errorf("line %d: expected 3 tab-separated fields, got %d", lineNum, len(parts))
		}

		word := parts[0]
		reading := parts[1]
		phonemeStrs := strings.Fields(parts[2])

		phonemes := make([]acoustic.Phoneme, len(phonemeStrs))
		for i, p := range phonemeStrs {
			phonemes[i] = acoustic.Phoneme(p)
		}

		d.Add(word, reading, phonemes)
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return d, nil
}

// LoadFile is a convenience wrapper that opens a file path.
func LoadFile(path string) (*Dictionary, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return Load(f)
}

// Lookup returns all pronunciation variants for a word.
func (d *Dictionary) Lookup(word string) []Entry {
	return d.Entries[word]
}

// PhonemeSequence returns the phoneme sequence for a word (first pronunciation).
func (d *Dictionary) PhonemeSequence(word string) ([]acoustic.Phoneme, bool) {
	entries := d.Entries[word]
	if len(entries) == 0 {
		return nil, false
	}
	return entries[0].Phonemes, true
}

// Words returns all words in the dictionary.
func (d *Dictionary) Words() []string {
	words := make([]string, 0, len(d.Entries))
	for w := range d.Entries {
		words = append(words, w)
	}
	return words
}
