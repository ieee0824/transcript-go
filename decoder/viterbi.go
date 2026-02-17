package decoder

import (
	"math"
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

// silWord is the special silence word that can appear between real words.
const silWord = "<sil>"

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

	// Locate silence word index (-1 if not in dictionary)
	silWordIdx := -1
	for wi, info := range wordInfos {
		if info.word == silWord {
			silWordIdx = wi
			break
		}
	}

	// Pre-compute emission log-likelihoods for all phonemes/states/frames.
	allPhonemes := acoustic.AllPhonemes()
	phonOrd := make(map[acoustic.Phoneme]int, len(allPhonemes))
	for i, ph := range allPhonemes {
		phonOrd[ph] = i
	}
	emitCols := len(allPhonemes) * acoustic.NumEmittingStates
	emitCache := mathutil.NewMat(T, emitCols)
	for pi, ph := range allPhonemes {
		hmm, ok := am.Phonemes[ph]
		if !ok {
			continue
		}
		for s := 1; s <= acoustic.NumEmittingStates; s++ {
			col := pi*acoustic.NumEmittingStates + (s - 1)
			gmm := hmm.States[s].GMM
			for t := 0; t < T; t++ {
				emitCache[t][col] = gmm.LogProb(features[t])
			}
		}
	}
	cachedEmit := func(ph acoustic.Phoneme, stateIdx, t int) float64 {
		return emitCache[t][phonOrd[ph]*acoustic.NumEmittingStates+(stateIdx-1)]
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

	// Initialize tokens: silence gets no LM score / penalty
	for wi, info := range wordInfos {
		phoneme := info.phonemes[0]
		if _, ok := am.Phonemes[phoneme]; !ok {
			continue
		}
		acScore := cachedEmit(phoneme, 1, 0)
		tok := currentPool.get()
		if wi == silWordIdx {
			tok.score = acScore
		} else {
			lmScore := lm.LogProb([]string{"<s>"}, info.word) * cfg.LMWeight
			tok.score = acScore + lmScore + cfg.WordInsertionPenalty
		}
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
		bestNextScore := math.Inf(-1)

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
				acScore := cachedEmit(phoneme, tok.stateIdx, t)
				s := tok.score + selfTrans + acScore
				nt := nextPool.get()
				nt.score = s
				nt.wordIdx = tok.wordIdx
				nt.phonIdx = tok.phonIdx
				nt.stateIdx = tok.stateIdx
				nt.history = tok.history
				nextTokens = append(nextTokens, nt)
				if s > bestNextScore {
					bestNextScore = s
				}
			}

			// Forward transition within phoneme HMM
			nextState := tok.stateIdx + 1
			if nextState <= acoustic.NumEmittingStates {
				fwdTrans := hmm.TransLog[tok.stateIdx][nextState]
				if fwdTrans > mathutil.LogZero+1 {
					acScore := cachedEmit(phoneme, nextState, t)
					s := tok.score + fwdTrans + acScore
					nt := nextPool.get()
					nt.score = s
					nt.wordIdx = tok.wordIdx
					nt.phonIdx = tok.phonIdx
					nt.stateIdx = nextState
					nt.history = tok.history
					nextTokens = append(nextTokens, nt)
					if s > bestNextScore {
						bestNextScore = s
					}
				}
			}

			// Transition to exit state -> next phoneme or word boundary
			// Apply a floor of log(0.5) when TransLog[s][exit] is LogZero.
			exitState := acoustic.NumStatesPerPhoneme - 1
			exitTrans := hmm.TransLog[tok.stateIdx][exitState]
			if exitTrans <= mathutil.LogZero+1 {
				exitTrans = math.Log(0.5)
			}
			if exitTrans > mathutil.LogZero+1 {
				if tok.phonIdx+1 < len(info.phonemes) {
					// Next phoneme in current word
					nextPhon := info.phonemes[tok.phonIdx+1]
					if _, ok := am.Phonemes[nextPhon]; ok {
						acScore := cachedEmit(nextPhon, 1, t)
						s := tok.score + exitTrans + acScore
						nt := nextPool.get()
						nt.score = s
						nt.wordIdx = tok.wordIdx
						nt.phonIdx = tok.phonIdx + 1
						nt.stateIdx = 1
						nt.history = tok.history
						nextTokens = append(nextTokens, nt)
						if s > bestNextScore {
							bestNextScore = s
						}
					}
				} else {
					// Word boundary
					baseScore := tok.score + exitTrans
					if bestNextScore > math.Inf(-1) && baseScore < bestNextScore-cfg.BeamWidth {
						continue
					}

					if silWordIdx >= 0 && tok.wordIdx == silWordIdx {
						// === Silence completed ===
						// Don't push sil to history; use pre-sil LM context.
						// Build LM history from existing (pre-sil) history.
						lmHistBuf = lmHistBuf[:0]
						cur := tok.history
						for cur != nil {
							lmHistBuf = append(lmHistBuf, cur.word)
							cur = cur.prev
						}
						for i, j := 0, len(lmHistBuf)-1; i < j; i, j = i+1, j-1 {
							lmHistBuf[i], lmHistBuf[j] = lmHistBuf[j], lmHistBuf[i]
						}

						// Expand to all non-silence words
						for nwi, ninfo := range wordInfos {
							if nwi == silWordIdx {
								continue // prevent sil→sil
							}
							nextPhon := ninfo.phonemes[0]
							if _, ok := am.Phonemes[nextPhon]; !ok {
								continue
							}
							var lmScore float64
							if len(lmHistBuf) > 0 {
								lmScore = lm.LogProb(lmHistBuf, ninfo.word) * cfg.LMWeight
							} else {
								lmScore = lm.LogProb([]string{"<s>"}, ninfo.word) * cfg.LMWeight
							}
							acScore := cachedEmit(nextPhon, 1, t)
							s := baseScore + lmScore + acScore + cfg.WordInsertionPenalty
							nt := nextPool.get()
							nt.score = s
							nt.wordIdx = nwi
							nt.phonIdx = 0
							nt.stateIdx = 1
							nt.history = tok.history // pass through, don't push sil
							nextTokens = append(nextTokens, nt)
							if s > bestNextScore {
								bestNextScore = s
							}
						}
					} else {
						// === Regular word completed ===
						newNode := &wordHistoryNode{
							word:  info.word,
							frame: t,
							prev:  tok.history,
						}
						if tok.history != nil {
							newNode.length = tok.history.length + 1
						} else {
							newNode.length = 1
						}

						// Build LM history
						lmHistBuf = lmHistBuf[:0]
						cur := tok.history
						for cur != nil {
							lmHistBuf = append(lmHistBuf, cur.word)
							cur = cur.prev
						}
						for i, j := 0, len(lmHistBuf)-1; i < j; i, j = i+1, j-1 {
							lmHistBuf[i], lmHistBuf[j] = lmHistBuf[j], lmHistBuf[i]
						}
						lmHistBuf = append(lmHistBuf, info.word)

						for nwi, ninfo := range wordInfos {
							nextPhon := ninfo.phonemes[0]
							if _, ok := am.Phonemes[nextPhon]; !ok {
								continue
							}
							acScore := cachedEmit(nextPhon, 1, t)
							if silWordIdx >= 0 && nwi == silWordIdx {
								// Transition to silence: acoustic only, no LM / penalty
								s := baseScore + acScore
								nt := nextPool.get()
								nt.score = s
								nt.wordIdx = nwi
								nt.phonIdx = 0
								nt.stateIdx = 1
								nt.history = newNode
								nextTokens = append(nextTokens, nt)
								if s > bestNextScore {
									bestNextScore = s
								}
							} else {
								lmScore := lm.LogProb(lmHistBuf, ninfo.word) * cfg.LMWeight
								s := baseScore + lmScore + acScore + cfg.WordInsertionPenalty
								nt := nextPool.get()
								nt.score = s
								nt.wordIdx = nwi
								nt.phonIdx = 0
								nt.stateIdx = 1
								nt.history = newNode
								nextTokens = append(nextTokens, nt)
								if s > bestNextScore {
									bestNextScore = s
								}
							}
						}
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

	// Build result: if final word is silence, use history as-is
	var finalNode *wordHistoryNode
	if silWordIdx >= 0 && best.wordIdx == silWordIdx {
		finalNode = best.history
	} else {
		finalNode = &wordHistoryNode{
			word: wordInfos[best.wordIdx].word,
			prev: best.history,
		}
		if best.history != nil {
			finalNode.length = best.history.length + 1
		} else {
			finalNode.length = 1
		}
	}

	if finalNode == nil {
		return &Result{LogScore: best.score}
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
