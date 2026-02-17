package acoustic

import "fmt"

// Triphone represents a context-dependent phoneme in "left-center+right" format.
// Word boundaries are represented by "#".
// Example: for word [i, k, u] → triphones are "#-i+k", "i-k+u", "k-u+#".
type Triphone string

// WordBoundary is the context symbol for word edges.
const WordBoundary = "#"

// MakeTriphone constructs a triphone string from its components.
func MakeTriphone(left, center, right string) Triphone {
	return Triphone(fmt.Sprintf("%s-%s+%s", left, center, right))
}

// CenterPhoneme extracts the center (base) phoneme from a triphone.
// For "i-k+u", returns Phoneme("k").
func (t Triphone) CenterPhoneme() Phoneme {
	s := string(t)
	dashIdx := -1
	plusIdx := -1
	for i := 0; i < len(s); i++ {
		if s[i] == '-' && dashIdx == -1 {
			dashIdx = i
		}
		if s[i] == '+' {
			plusIdx = i
		}
	}
	if dashIdx >= 0 && plusIdx > dashIdx {
		return Phoneme(s[dashIdx+1 : plusIdx])
	}
	return Phoneme(s) // fallback: treat as monophone
}

// WordToTriphones converts a word's monophone sequence to word-internal triphones.
// Uses "#" for word boundary context.
// Example: [i, k, u] → [#-i+k, i-k+u, k-u+#]
// Single phoneme: [a] → [#-a+#]
func WordToTriphones(phonemes []Phoneme) []Triphone {
	N := len(phonemes)
	if N == 0 {
		return nil
	}
	triphones := make([]Triphone, N)
	for i := 0; i < N; i++ {
		var left, right string
		if i == 0 {
			left = WordBoundary
		} else {
			left = string(phonemes[i-1])
		}
		if i == N-1 {
			right = WordBoundary
		} else {
			right = string(phonemes[i+1])
		}
		triphones[i] = MakeTriphone(left, string(phonemes[i]), right)
	}
	return triphones
}
