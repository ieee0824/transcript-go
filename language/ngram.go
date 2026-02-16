package language

import "github.com/ieee0824/transcript-go/internal/mathutil"

// NGramModel represents an n-gram language model.
type NGramModel struct {
	Order    int                    // 2 for bigram, 3 for trigram
	Unigrams map[string]ngramEntry  // word -> entry
	Bigrams  map[[2]string]ngramEntry
	Trigrams map[[3]string]ngramEntry
}

type ngramEntry struct {
	LogProb   float64
	LogBackoff float64
}

// NewNGramModel creates an empty n-gram model.
func NewNGramModel(order int) *NGramModel {
	return &NGramModel{
		Order:    order,
		Unigrams: make(map[string]ngramEntry),
		Bigrams:  make(map[[2]string]ngramEntry),
		Trigrams: make(map[[3]string]ngramEntry),
	}
}

// LogProb returns the log probability of a word given its history.
// Uses backoff when the exact n-gram is not found.
func (m *NGramModel) LogProb(history []string, word string) float64 {
	if m.Order >= 3 && len(history) >= 2 {
		key := [3]string{history[len(history)-2], history[len(history)-1], word}
		if e, ok := m.Trigrams[key]; ok {
			return e.LogProb
		}
		// Backoff to bigram
		biKey := [2]string{history[len(history)-2], history[len(history)-1]}
		if e, ok := m.Bigrams[biKey]; ok {
			return e.LogBackoff + m.logProbBigram(history[len(history)-1], word)
		}
	}

	if m.Order >= 2 && len(history) >= 1 {
		return m.logProbBigram(history[len(history)-1], word)
	}

	return m.logProbUnigram(word)
}

func (m *NGramModel) logProbBigram(prev, word string) float64 {
	key := [2]string{prev, word}
	if e, ok := m.Bigrams[key]; ok {
		return e.LogProb
	}
	// Backoff to unigram
	if e, ok := m.Unigrams[prev]; ok {
		return e.LogBackoff + m.logProbUnigram(word)
	}
	return m.logProbUnigram(word)
}

func (m *NGramModel) logProbUnigram(word string) float64 {
	if e, ok := m.Unigrams[word]; ok {
		return e.LogProb
	}
	return mathutil.LogZero
}

// SentenceLogProb returns the total log probability of a sentence (word sequence).
// Automatically adds <s> at the beginning and </s> at the end.
func (m *NGramModel) SentenceLogProb(words []string) float64 {
	total := 0.0
	history := []string{"<s>"}
	for _, w := range words {
		total += m.LogProb(history, w)
		history = append(history, w)
	}
	total += m.LogProb(history, "</s>")
	return total
}

// Vocab returns all words in the unigram vocabulary.
func (m *NGramModel) Vocab() []string {
	words := make([]string, 0, len(m.Unigrams))
	for w := range m.Unigrams {
		words = append(words, w)
	}
	return words
}
