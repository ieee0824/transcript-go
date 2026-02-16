package decoder

import (
	"sort"
	"strings"
	"github.com/ieee0824/transcript-go/acoustic"
	"github.com/ieee0824/transcript-go/internal/mathutil"
	"github.com/ieee0824/transcript-go/language"
	"github.com/ieee0824/transcript-go/lexicon"
)

// Config holds beam search parameters.
type Config struct {
	BeamWidth            float64 // log-domain beam width
	MaxActiveTokens      int     // maximum number of active hypotheses
	LMWeight             float64 // language model scaling factor
	WordInsertionPenalty float64 // penalty to control insertion rate
}

// DefaultConfig returns reasonable default parameters.
func DefaultConfig() Config {
	return Config{
		BeamWidth:            200.0,
		MaxActiveTokens:      1000,
		LMWeight:             10.0,
		WordInsertionPenalty: 0.0,
	}
}

// wordHistoryNode is a linked list node to avoid copying word history slices.
type wordHistoryNode struct {
	word   string
	frame  int // start frame of this word
	prev   *wordHistoryNode
	length int
}

func (n *wordHistoryNode) toSlice() ([]string, []int) {
	if n == nil {
		return nil, nil
	}
	words := make([]string, n.length)
	frames := make([]int, n.length)
	cur := n
	for i := n.length - 1; i >= 0; i-- {
		words[i] = cur.word
		frames[i] = cur.frame
		cur = cur.prev
	}
	return words, frames
}

// token represents an active hypothesis in beam search.
type token struct {
	score    float64
	wordIdx  int
	phonIdx  int
	stateIdx int
	history  *wordHistoryNode
}

// tokenPool manages pre-allocated token slices to reduce allocations.
type tokenPool struct {
	buf []token
	pos int
}

func newTokenPool(cap int) *tokenPool {
	return &tokenPool{buf: make([]token, cap), pos: 0}
}

func (p *tokenPool) get() *token {
	if p.pos >= len(p.buf) {
		// Grow
		p.buf = append(p.buf, make([]token, len(p.buf))...)
	}
	t := &p.buf[p.pos]
	p.pos++
	return t
}

func (p *tokenPool) reset() {
	p.pos = 0
}

// Decode performs Viterbi beam search decoding on a sequence of feature frames.
func Decode(features [][]float64, am *acoustic.AcousticModel, lm *language.NGramModel, dict *lexicon.Dictionary, cfg Config) *Result {
	T := len(features)
	if T == 0 {
		return &Result{}
	}

	vocab := dict.Words()
	if len(vocab) == 0 {
		return &Result{}
	}

	// Build phoneme sequences for each word
	type wordInfo struct {
		word     string
		phonemes []acoustic.Phoneme
	}
	wordInfos := make([]wordInfo, 0, len(vocab))
	for _, w := range vocab {
		phonemes, ok := dict.PhonemeSequence(w)
		if !ok || len(phonemes) == 0 {
			continue
		}
		wordInfos = append(wordInfos, wordInfo{word: w, phonemes: phonemes})
	}

	if len(wordInfos) == 0 {
		return &Result{}
	}

	// Pre-allocate token pools (double-buffer)
	estimatedTokens := len(wordInfos) * 4
	if estimatedTokens < 256 {
		estimatedTokens = 256
	}
	pool1 := newTokenPool(estimatedTokens)
	pool2 := newTokenPool(estimatedTokens)
	currentPool := pool1
	nextPool := pool2

	// Pre-allocate pointer slices
	activeTokens := make([]*token, 0, estimatedTokens)
	nextTokens := make([]*token, 0, estimatedTokens*2)

	// Reusable LM history buffer
	lmHistBuf := make([]string, 0, 16)

	// Initialize tokens
	for wi, info := range wordInfos {
		phoneme := info.phonemes[0]
		hmm, ok := am.Phonemes[phoneme]
		if !ok {
			continue
		}
		lmScore := lm.LogProb([]string{"<s>"}, info.word) * cfg.LMWeight
		acScore := hmm.LogLikelihood(1, features[0])
		tok := currentPool.get()
		tok.score = acScore + lmScore + cfg.WordInsertionPenalty
		tok.wordIdx = wi
		tok.phonIdx = 0
		tok.stateIdx = 1
		tok.history = nil
		activeTokens = append(activeTokens, tok)
	}

	// Frame-synchronous beam search
	for t := 1; t < T; t++ {
		nextPool.reset()
		nextTokens = nextTokens[:0]

		for _, tok := range activeTokens {
			info := wordInfos[tok.wordIdx]
			phoneme := info.phonemes[tok.phonIdx]
			hmm, ok := am.Phonemes[phoneme]
			if !ok {
				continue
			}

			// Self-loop: stay in same state
			selfTrans := hmm.TransLog[tok.stateIdx][tok.stateIdx]
			if selfTrans > mathutil.LogZero+1 {
				acScore := hmm.LogLikelihood(tok.stateIdx, features[t])
				nt := nextPool.get()
				nt.score = tok.score + selfTrans + acScore
				nt.wordIdx = tok.wordIdx
				nt.phonIdx = tok.phonIdx
				nt.stateIdx = tok.stateIdx
				nt.history = tok.history
				nextTokens = append(nextTokens, nt)
			}

			// Forward transition within phoneme HMM
			nextState := tok.stateIdx + 1
			if nextState <= acoustic.NumEmittingStates {
				fwdTrans := hmm.TransLog[tok.stateIdx][nextState]
				if fwdTrans > mathutil.LogZero+1 {
					acScore := hmm.LogLikelihood(nextState, features[t])
					nt := nextPool.get()
					nt.score = tok.score + fwdTrans + acScore
					nt.wordIdx = tok.wordIdx
					nt.phonIdx = tok.phonIdx
					nt.stateIdx = nextState
					nt.history = tok.history
					nextTokens = append(nextTokens, nt)
				}
			}

			// Transition to exit state -> next phoneme or word boundary
			exitState := acoustic.NumStatesPerPhoneme - 1
			exitTrans := hmm.TransLog[tok.stateIdx][exitState]
			if exitTrans > mathutil.LogZero+1 {
				if tok.phonIdx+1 < len(info.phonemes) {
					// Next phoneme in current word
					nextPhon := info.phonemes[tok.phonIdx+1]
					nextHMM, ok := am.Phonemes[nextPhon]
					if ok {
						acScore := nextHMM.LogLikelihood(1, features[t])
						nt := nextPool.get()
						nt.score = tok.score + exitTrans + acScore
						nt.wordIdx = tok.wordIdx
						nt.phonIdx = tok.phonIdx + 1
						nt.stateIdx = 1
						nt.history = tok.history
						nextTokens = append(nextTokens, nt)
					}
				} else {
					// Word boundary: push current word to history
					newNode := &wordHistoryNode{
						word:  info.word,
						frame: 0,
						prev:  tok.history,
					}
					if tok.history != nil {
						newNode.length = tok.history.length + 1
					} else {
						newNode.length = 1
					}
					// Recover start frame from the current token context
					if tok.history != nil {
						// not trivial to know exact start, use t as word-end marker
					}

					// Build LM history from linked list
					lmHistBuf = lmHistBuf[:0]
					cur := tok.history
					for cur != nil {
						lmHistBuf = append(lmHistBuf, cur.word)
						cur = cur.prev
					}
					// Reverse to get chronological order, then append current word
					for i, j := 0, len(lmHistBuf)-1; i < j; i, j = i+1, j-1 {
						lmHistBuf[i], lmHistBuf[j] = lmHistBuf[j], lmHistBuf[i]
					}
					lmHistBuf = append(lmHistBuf, info.word)

					baseScore := tok.score + exitTrans
					for nwi, ninfo := range wordInfos {
						nextPhon := ninfo.phonemes[0]
						nextHMM, ok := am.Phonemes[nextPhon]
						if !ok {
							continue
						}
						lmScore := lm.LogProb(lmHistBuf, ninfo.word) * cfg.LMWeight
						acScore := nextHMM.LogLikelihood(1, features[t])
						nt := nextPool.get()
						nt.score = baseScore + lmScore + acScore + cfg.WordInsertionPenalty
						nt.wordIdx = nwi
						nt.phonIdx = 0
						nt.stateIdx = 1
						nt.history = newNode
						nextTokens = append(nextTokens, nt)
					}
				}
			}
		}

		// Beam pruning (in-place into activeTokens)
		activeTokens = pruneTokens(nextTokens, activeTokens[:0], cfg.BeamWidth, cfg.MaxActiveTokens)

		// Swap pools
		currentPool, nextPool = nextPool, currentPool
	}

	// Find best token
	if len(activeTokens) == 0 {
		return &Result{}
	}

	best := activeTokens[0]
	for _, tok := range activeTokens[1:] {
		if tok.score > best.score {
			best = tok
		}
	}

	// Build result: collect final word (current wordIdx) + history
	finalNode := &wordHistoryNode{
		word: wordInfos[best.wordIdx].word,
		prev: best.history,
	}
	if best.history != nil {
		finalNode.length = best.history.length + 1
	} else {
		finalNode.length = 1
	}

	words, starts := finalNode.toSlice()
	result := &Result{
		Text:     strings.Join(words, ""),
		LogScore: best.score,
	}

	for i, w := range words {
		word := Word{Text: w}
		if i < len(starts) {
			word.StartFrame = starts[i]
		}
		if i+1 < len(starts) {
			word.EndFrame = starts[i+1] - 1
		} else {
			word.EndFrame = T - 1
		}
		result.Words = append(result.Words, word)
	}

	return result
}

func pruneTokens(src []*token, dst []*token, beamWidth float64, maxActive int) []*token {
	if len(src) == 0 {
		return dst
	}

	// Find best score
	bestScore := src[0].score
	for _, tok := range src[1:] {
		if tok.score > bestScore {
			bestScore = tok.score
		}
	}

	// Beam pruning: reuse dst slice
	threshold := bestScore - beamWidth
	for _, tok := range src {
		if tok.score >= threshold {
			dst = append(dst, tok)
		}
	}

	// Max active pruning
	if len(dst) > maxActive {
		sort.Slice(dst, func(i, j int) bool {
			return dst[i].score > dst[j].score
		})
		dst = dst[:maxActive]
	}

	return dst
}
