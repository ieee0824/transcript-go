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

	// Build phoneme sequences and resolve HMMs for each word
	type wordInfo struct {
		word     string
		phonemes []acoustic.Phoneme
		hmms     []*acoustic.PhonemeHMM // resolved HMM per phoneme position
		hmmOrds  []int                  // index into uniqueHMMs for emission cache
	}
	wordInfos := make([]wordInfo, 0, len(vocab))
	for _, w := range vocab {
		phonemes, ok := dict.PhonemeSequence(w)
		if !ok || len(phonemes) == 0 {
			continue
		}
		info := wordInfo{word: w, phonemes: phonemes}
		// Resolve HMMs: triphone if available, else monophone
		if am.HasTriphones() {
			triphones := acoustic.WordToTriphones(phonemes)
			info.hmms = make([]*acoustic.PhonemeHMM, len(phonemes))
			for i, tri := range triphones {
				info.hmms[i] = am.ResolveHMM(tri)
			}
		} else {
			info.hmms = make([]*acoustic.PhonemeHMM, len(phonemes))
			for i, ph := range phonemes {
				info.hmms[i] = am.Phonemes[ph]
			}
		}
		// Check first phoneme has valid HMM
		if info.hmms[0] == nil {
			continue
		}
		wordInfos = append(wordInfos, info)
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

	// Build unique HMM set for emission cache
	hmmSet := make(map[*acoustic.PhonemeHMM]int)
	var uniqueHMMs []*acoustic.PhonemeHMM
	for i := range wordInfos {
		wordInfos[i].hmmOrds = make([]int, len(wordInfos[i].phonemes))
		for j, hmm := range wordInfos[i].hmms {
			ord, exists := hmmSet[hmm]
			if !exists {
				ord = len(uniqueHMMs)
				hmmSet[hmm] = ord
				uniqueHMMs = append(uniqueHMMs, hmm)
			}
			wordInfos[i].hmmOrds[j] = ord
		}
	}

	// Pre-compute emission log-likelihoods for all unique HMMs
	emitCols := len(uniqueHMMs) * acoustic.NumEmittingStates
	emitCache := mathutil.NewMat(T, emitCols)
	for hi, hmm := range uniqueHMMs {
		for s := 1; s <= acoustic.NumEmittingStates; s++ {
			col := hi*acoustic.NumEmittingStates + (s - 1)
			gmm := hmm.States[s].GMM
			for t := 0; t < T; t++ {
				emitCache[t][col] = gmm.LogProb(features[t])
			}
		}
	}
	cachedEmit := func(hmmOrd int, stateIdx, t int) float64 {
		return emitCache[t][hmmOrd*acoustic.NumEmittingStates+(stateIdx-1)]
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
		if info.hmms[0] == nil {
			continue
		}
		acScore := cachedEmit(info.hmmOrds[0], 1, 0)
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
			hmm := info.hmms[tok.phonIdx]
			hmmOrd := info.hmmOrds[tok.phonIdx]

			// Self-loop: stay in same state
			selfTrans := hmm.TransLog[tok.stateIdx][tok.stateIdx]
			if selfTrans > mathutil.LogZero+1 {
				acScore := cachedEmit(hmmOrd, tok.stateIdx, t)
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
					acScore := cachedEmit(hmmOrd, nextState, t)
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
			exitState := acoustic.NumStatesPerPhoneme - 1
			exitTrans := hmm.TransLog[tok.stateIdx][exitState]
			if exitTrans <= mathutil.LogZero+1 {
				exitTrans = math.Log(0.5)
			}
			if exitTrans > mathutil.LogZero+1 {
				if tok.phonIdx+1 < len(info.phonemes) {
					// Next phoneme in current word
					nextHmmOrd := info.hmmOrds[tok.phonIdx+1]
					acScore := cachedEmit(nextHmmOrd, 1, t)
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
				} else {
					// Word boundary
					baseScore := tok.score + exitTrans
					if bestNextScore > math.Inf(-1) && baseScore < bestNextScore-cfg.BeamWidth {
						continue
					}

					if silWordIdx >= 0 && tok.wordIdx == silWordIdx {
						// === Silence completed ===
						lmHistBuf = lmHistBuf[:0]
						cur := tok.history
						for cur != nil {
							lmHistBuf = append(lmHistBuf, cur.word)
							cur = cur.prev
						}
						for i, j := 0, len(lmHistBuf)-1; i < j; i, j = i+1, j-1 {
							lmHistBuf[i], lmHistBuf[j] = lmHistBuf[j], lmHistBuf[i]
						}

						for nwi, ninfo := range wordInfos {
							if nwi == silWordIdx {
								continue
							}
							if ninfo.hmms[0] == nil {
								continue
							}
							var lmScore float64
							if len(lmHistBuf) > 0 {
								lmScore = lm.LogProb(lmHistBuf, ninfo.word) * cfg.LMWeight
							} else {
								lmScore = lm.LogProb([]string{"<s>"}, ninfo.word) * cfg.LMWeight
							}
							acScore := cachedEmit(ninfo.hmmOrds[0], 1, t)
							s := baseScore + lmScore + acScore + cfg.WordInsertionPenalty
							nt := nextPool.get()
							nt.score = s
							nt.wordIdx = nwi
							nt.phonIdx = 0
							nt.stateIdx = 1
							nt.history = tok.history
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
							if ninfo.hmms[0] == nil {
								continue
							}
							acScore := cachedEmit(ninfo.hmmOrds[0], 1, t)
							if silWordIdx >= 0 && nwi == silWordIdx {
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
