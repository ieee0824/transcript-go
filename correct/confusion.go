package correct

import (
	"sort"
	"strings"

	"github.com/ieee0824/transcript-go/acoustic"
	"github.com/ieee0824/transcript-go/lexicon"
)

// Candidate represents a correction candidate for a word.
type Candidate struct {
	Word     string
	Phonemes []acoustic.Phoneme
	Distance int     // phoneme edit distance from input word
	Score    float64 // confusion score (0 for exact match, negative for distance)
}

// ConfusionModel maps pronunciations to candidate words.
type ConfusionModel struct {
	// pronunciation (phoneme string key) → list of entries
	byPronunciation map[string][]pronEntry
	// word → first phoneme sequence (for looking up input words)
	wordPhonemes map[string][]acoustic.Phoneme
	// all unique pronunciation keys for edit distance search
	allPronKeys []string
	allPronPhon [][]acoustic.Phoneme
}

type pronEntry struct {
	word     string
	phonemes []acoustic.Phoneme
}

// phonemeKey creates a string key from a phoneme slice for map lookup.
func phonemeKey(phonemes []acoustic.Phoneme) string {
	parts := make([]string, len(phonemes))
	for i, p := range phonemes {
		parts[i] = string(p)
	}
	return strings.Join(parts, " ")
}

// BuildConfusionModel builds a confusion model from a large dictionary.
// smallDict is the decoder's dictionary (used to look up input word pronunciations).
// largeDict provides the candidate words for correction.
func BuildConfusionModel(largeDict, smallDict *lexicon.Dictionary) *ConfusionModel {
	m := &ConfusionModel{
		byPronunciation: make(map[string][]pronEntry),
		wordPhonemes:    make(map[string][]acoustic.Phoneme),
	}

	// Index all words from smallDict for input word lookup
	for word, entries := range smallDict.Entries {
		if len(entries) > 0 {
			m.wordPhonemes[word] = entries[0].Phonemes
		}
	}

	// Index all words from largeDict by pronunciation
	seen := make(map[string]map[string]bool) // pronKey → set of words already added
	for word, entries := range largeDict.Entries {
		for _, e := range entries {
			key := phonemeKey(e.Phonemes)
			if seen[key] == nil {
				seen[key] = make(map[string]bool)
			}
			if seen[key][word] {
				continue
			}
			seen[key][word] = true
			m.byPronunciation[key] = append(m.byPronunciation[key], pronEntry{
				word:     word,
				phonemes: e.Phonemes,
			})
		}
	}

	// Build list of unique pronunciation keys for edit distance search
	for key, entries := range m.byPronunciation {
		if len(entries) > 0 {
			m.allPronKeys = append(m.allPronKeys, key)
			m.allPronPhon = append(m.allPronPhon, entries[0].phonemes)
		}
	}

	return m
}

// Candidates returns correction candidates for a word.
// It returns exact pronunciation matches (distance 0) and
// near matches within maxDist phoneme edit distance.
// The input word itself is always included as a candidate with distance 0.
func (m *ConfusionModel) Candidates(word string, maxDist int) []Candidate {
	phonemes, ok := m.wordPhonemes[word]
	if !ok {
		// Unknown word: return only itself
		return []Candidate{{Word: word, Distance: 0, Score: 0}}
	}

	inputKey := phonemeKey(phonemes)
	candidateMap := make(map[string]Candidate)

	// Always include the input word itself
	candidateMap[word] = Candidate{
		Word:     word,
		Phonemes: phonemes,
		Distance: 0,
		Score:    0,
	}

	// Tier 1: exact pronunciation matches (homophones)
	if entries, ok := m.byPronunciation[inputKey]; ok {
		for _, e := range entries {
			if _, exists := candidateMap[e.word]; !exists {
				candidateMap[e.word] = Candidate{
					Word:     e.word,
					Phonemes: e.phonemes,
					Distance: 0,
					Score:    0, // same pronunciation = no penalty
				}
			}
		}
	}

	// Tier 2: near pronunciation matches (edit distance <= maxDist)
	if maxDist > 0 {
		for i, pronPhon := range m.allPronPhon {
			dist := lexicon.PhonemeEditDistance(phonemes, pronPhon)
			if dist > 0 && dist <= maxDist {
				key := m.allPronKeys[i]
				for _, e := range m.byPronunciation[key] {
					if existing, exists := candidateMap[e.word]; !exists || dist < existing.Distance {
						candidateMap[e.word] = Candidate{
							Word:     e.word,
							Phonemes: e.phonemes,
							Distance: dist,
							Score:    -float64(dist * dist), // quadratic penalty
						}
					}
				}
			}
		}
	}

	candidates := make([]Candidate, 0, len(candidateMap))
	for _, c := range candidateMap {
		candidates = append(candidates, c)
	}
	// Sort for deterministic beam search results
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].Word < candidates[j].Word
	})
	return candidates
}
