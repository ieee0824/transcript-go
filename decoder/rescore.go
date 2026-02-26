package decoder

import (
	"strings"

	"github.com/ieee0824/transcript-go/language"
)

// RescoreNBest rescores N-best hypotheses using a language model.
// It computes a combined score: original + lmWeight * SentenceLogProb(words).
// Returns the best-scoring hypothesis after rescoring.
func RescoreNBest(nbest []Result, lm *language.NGramModel, lmWeight float64) *Result {
	if len(nbest) == 0 {
		return &Result{}
	}
	if lm == nil || lmWeight == 0 {
		r := nbest[0]
		return &r
	}

	bestIdx := 0
	bestScore := nbest[0].LogScore + lmWeight*lm.SentenceLogProb(strings.Fields(nbest[0].Text))

	for i := 1; i < len(nbest); i++ {
		words := strings.Fields(nbest[i].Text)
		score := nbest[i].LogScore + lmWeight*lm.SentenceLogProb(words)
		if score > bestScore {
			bestScore = score
			bestIdx = i
		}
	}

	r := nbest[bestIdx]
	r.LogScore = bestScore
	return &r
}
