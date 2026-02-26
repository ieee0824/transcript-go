package correct

import (
	"math"
	"sort"
	"strings"

	"github.com/ieee0824/transcript-go/language"
)

// Corrector performs text-to-text error correction using a noisy channel model.
type Corrector struct {
	LM              *language.NGramModel
	Confusion       *ConfusionModel
	LMWeight        float64 // weight for LM score (default: 1.0)
	ConfusionWeight float64 // weight for confusion score (default: 1.0)
	BeamWidth       int     // beam width for Viterbi search (default: 50)
	MaxDist         int     // max phoneme edit distance for candidates (default: 2)
	KeepBonus       float64 // bonus for keeping the original input word (default: 0)
}

// hypothesis represents a partial correction hypothesis in beam search.
type hypothesis struct {
	words []string // corrected word sequence so far
	score float64  // accumulated score
	// last two words for trigram context (empty string = sentence start)
	prev     string
	prevPrev string
}

// Correct takes decoder output words and returns corrected words.
func (c *Corrector) Correct(words []string) []string {
	if len(words) == 0 {
		return words
	}

	beamWidth := c.BeamWidth
	if beamWidth <= 0 {
		beamWidth = 50
	}
	maxDist := c.MaxDist
	if maxDist < 0 {
		maxDist = 2
	}
	lmWeight := c.LMWeight
	if lmWeight == 0 {
		lmWeight = 1.0
	}
	confWeight := c.ConfusionWeight
	if confWeight == 0 {
		confWeight = 1.0
	}

	// Initialize beam with sentence-start hypothesis
	beam := []hypothesis{{
		words:    nil,
		score:    0,
		prev:     "<s>",
		prevPrev: "",
	}}

	// Process each word position
	for _, inputWord := range words {
		candidates := c.Confusion.Candidates(inputWord, maxDist)
		if len(candidates) == 0 {
			// No candidates: keep original word
			candidates = []Candidate{{Word: inputWord, Distance: 0, Score: 0}}
		}

		var nextBeam []hypothesis
		for _, hyp := range beam {
			for _, cand := range candidates {
				// LM score: P(candidate | context)
				var history []string
				if hyp.prevPrev != "" {
					history = []string{hyp.prevPrev, hyp.prev}
				} else if hyp.prev != "" {
					history = []string{hyp.prev}
				}
				lmScore := c.LM.LogProb(history, cand.Word)

				// Combined score
				candScore := cand.Score
				if cand.Word == inputWord {
					candScore += c.KeepBonus
				}
				score := hyp.score + lmWeight*lmScore + confWeight*candScore

				newWords := make([]string, len(hyp.words)+1)
				copy(newWords, hyp.words)
				newWords[len(hyp.words)] = cand.Word

				nextBeam = append(nextBeam, hypothesis{
					words:    newWords,
					score:    score,
					prev:     cand.Word,
					prevPrev: hyp.prev,
				})
			}
		}

		// Prune: keep top-K hypotheses
		beam = pruneBeam(nextBeam, beamWidth)
	}

	// Add sentence-end score
	for i := range beam {
		var history []string
		if beam[i].prevPrev != "" {
			history = []string{beam[i].prevPrev, beam[i].prev}
		} else if beam[i].prev != "" {
			history = []string{beam[i].prev}
		}
		beam[i].score += lmWeight * c.LM.LogProb(history, "</s>")
	}

	// Find best hypothesis
	bestIdx := 0
	bestScore := math.Inf(-1)
	for i, hyp := range beam {
		if hyp.score > bestScore {
			bestScore = hyp.score
			bestIdx = i
		}
	}

	if len(beam) > 0 {
		return beam[bestIdx].words
	}
	return words
}

// CorrectText takes a space-separated string and returns corrected text.
func (c *Corrector) CorrectText(text string) string {
	words := strings.Fields(text)
	corrected := c.Correct(words)
	return strings.Join(corrected, " ")
}

// pruneBeam keeps only the top-K hypotheses by score.
func pruneBeam(beam []hypothesis, k int) []hypothesis {
	if len(beam) <= k {
		return beam
	}
	sort.Slice(beam, func(i, j int) bool {
		return beam[i].score > beam[j].score
	})
	return beam[:k]
}
