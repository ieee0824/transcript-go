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
	LMInterpolation      float64 // uniform interpolation weight (0=pure LM, 0.5=half LM half uniform)
}

// DefaultConfig returns reasonable default parameters.
func DefaultConfig() Config {
	return Config{
		BeamWidth:            200.0,
		MaxActiveTokens:      1000,
		LMWeight:             10.0,
		WordInsertionPenalty: 0.0,
		LMInterpolation:     0.0,
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

// trieNode represents a node in the phoneme prefix trie.
type trieNode struct {
	phoneme  acoustic.Phoneme
	hmm      *acoustic.PhonemeHMM
	hmmOrd   int // index into uniqueHMMs for emission cache
	children []trieChild
	wordEnds []string // words ending at this node
}

// trieChild is a (phoneme, nodeIdx) pair for trie children.
type trieChild struct {
	phoneme acoustic.Phoneme
	nodeIdx int
}

// token represents an active hypothesis in beam search.
type token struct {
	score    float64
	nodeIdx  int // trie node index; -1 = silence
	stateIdx int // HMM state (1-3)
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

// recomKey is used for token recombination within the trie.
type recomKey struct {
	nodeIdx  int
	stateIdx int
	lastWord string
}

// Decode performs Viterbi beam search decoding using a lexicon prefix trie.
func Decode(features [][]float64, am *acoustic.AcousticModel, lm *language.NGramModel, dict *lexicon.Dictionary, cfg Config) *Result {
	T := len(features)
	if T == 0 {
		return &Result{}
	}

	vocab := dict.Words()
	if len(vocab) == 0 {
		return &Result{}
	}

	// Build phoneme prefix trie from dictionary (monophone HMMs)
	nodes := []trieNode{{}} // nodes[0] = root (no phoneme, no HMM)
	hmmSet := make(map[*acoustic.PhonemeHMM]int)
	var uniqueHMMs []*acoustic.PhonemeHMM

	hasSil := false
	for _, w := range vocab {
		if w == silWord {
			hasSil = true
			continue
		}
		phonemes, ok := dict.PhonemeSequence(w)
		if !ok || len(phonemes) == 0 {
			continue
		}
		cur := 0
		for _, ph := range phonemes {
			found := -1
			for _, c := range nodes[cur].children {
				if c.phoneme == ph {
					found = c.nodeIdx
					break
				}
			}
			if found >= 0 {
				cur = found
			} else {
				hmm := am.Phonemes[ph]
				if hmm == nil {
					cur = -1
					break
				}
				ord, exists := hmmSet[hmm]
				if !exists {
					ord = len(uniqueHMMs)
					hmmSet[hmm] = ord
					uniqueHMMs = append(uniqueHMMs, hmm)
				}
				newIdx := len(nodes)
				nodes = append(nodes, trieNode{
					phoneme: ph,
					hmm:     hmm,
					hmmOrd:  ord,
				})
				nodes[cur].children = append(nodes[cur].children, trieChild{phoneme: ph, nodeIdx: newIdx})
				cur = newIdx
			}
		}
		if cur > 0 {
			nodes[cur].wordEnds = append(nodes[cur].wordEnds, w)
		}
	}

	// Resolve silence HMM
	var silHMM *acoustic.PhonemeHMM
	silHMMOrd := -1
	if hasSil {
		silPhons, ok := dict.PhonemeSequence(silWord)
		if ok && len(silPhons) > 0 {
			silHMM = am.Phonemes[silPhons[0]]
			if silHMM != nil {
				ord, exists := hmmSet[silHMM]
				if !exists {
					ord = len(uniqueHMMs)
					hmmSet[silHMM] = ord
					uniqueHMMs = append(uniqueHMMs, silHMM)
				}
				silHMMOrd = ord
			}
		}
	}

	if len(uniqueHMMs) == 0 {
		return &Result{}
	}

	rootChildren := nodes[0].children
	if len(rootChildren) == 0 && silHMM == nil {
		return &Result{}
	}

	// Compute LM look-ahead: best (maximum) unigram log probability.
	// Applied when entering the trie, corrected at word completion.
	// This prevents mid-trie tokens from having an unfair advantage over
	// completed-word tokens that have been penalized by LM scores.
	bestUniLM := math.Inf(-1)
	for _, v := range lm.Vocab() {
		s := lm.LogProb([]string{"<s>"}, v)
		if s > bestUniLM {
			bestUniLM = s
		}
	}
	if math.IsInf(bestUniLM, -1) {
		bestUniLM = 0
	}
	// Context-aware LM scoring: adaptive weight based on LM coverage.
	// - Word in LM vocab with known bigram context → full LMWeight
	// - Word in LM vocab but OOV context → LMWeight * (1-interpolation)
	// - OOV word → LMWeight * (1-interpolation)^2
	// When LMInterpolation=0, all cases use full LMWeight (original behavior).
	oovScale := 1.0 - cfg.LMInterpolation    // scale factor for OOV words
	ctxScale := 1.0 - cfg.LMInterpolation*0.5 // scale factor for known word with OOV context

	scoreLM := func(history []string, word string) float64 {
		rawLP := lm.LogProb(history, word)
		w := cfg.LMWeight
		if cfg.LMInterpolation > 0 {
			_, inVocab := lm.Unigrams[word]
			if !inVocab {
				// OOV word: LM has no knowledge, minimize its influence
				w *= oovScale
			} else if len(history) > 0 {
				lastWord := history[len(history)-1]
				if lastWord != "<s>" {
					if _, ok := lm.Unigrams[lastWord]; !ok {
						// Known word but OOV context: LM bigram unreliable
						w *= ctxScale
					}
				}
			}
		}
		return rawLP * w
	}

	// LM look-ahead: use full weight (maximum possible LM contribution)
	// This is correct because adaptive scaling only reduces the weight.
	bestUniLM = math.Inf(-1)
	for _, v := range lm.Vocab() {
		s := lm.LogProb([]string{"<s>"}, v)
		if s > bestUniLM {
			bestUniLM = s
		}
	}
	if math.IsInf(bestUniLM, -1) {
		bestUniLM = 0
	}
	lmLookAhead := bestUniLM * cfg.LMWeight

	// Pre-compute emission log-likelihoods for all unique monophone HMMs
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
	estimatedTokens := len(nodes) * 3
	if estimatedTokens < 256 {
		estimatedTokens = 256
	}
	pool1 := newTokenPool(estimatedTokens)
	pool2 := newTokenPool(estimatedTokens)
	currentPool := pool1
	nextPool := pool2

	activeTokens := make([]*token, 0, estimatedTokens)
	nextTokens := make([]*token, 0, estimatedTokens*2)
	lmHistBuf := make([]string, 0, 16)

	// Initialize tokens at t=0: root children (with look-ahead LM) + silence
	for _, c := range rootChildren {
		nd := &nodes[c.nodeIdx]
		acScore := cachedEmit(nd.hmmOrd, 1, 0)
		tok := currentPool.get()
		tok.score = acScore + lmLookAhead
		tok.nodeIdx = c.nodeIdx
		tok.stateIdx = 1
		tok.history = nil
		activeTokens = append(activeTokens, tok)
	}
	if silHMM != nil {
		tok := currentPool.get()
		tok.score = cachedEmit(silHMMOrd, 1, 0)
		tok.nodeIdx = -1
		tok.stateIdx = 1
		tok.history = nil
		activeTokens = append(activeTokens, tok)
	}

	// Token recombination map (reused each frame)
	recom := make(map[recomKey]int, estimatedTokens)

	// Frame-synchronous beam search
	for t := 1; t < T; t++ {
		nextPool.reset()
		nextTokens = nextTokens[:0]
		bestNextScore := math.Inf(-1)

		for k := range recom {
			delete(recom, k)
		}

		for _, tok := range activeTokens {
			if tok.nodeIdx == -1 {
				// === Silence token ===
				hmm := silHMM
				hmmOrd := silHMMOrd

				// Self-loop
				selfTrans := hmm.TransLog[tok.stateIdx][tok.stateIdx]
				if selfTrans > mathutil.LogZero+1 {
					acScore := cachedEmit(hmmOrd, tok.stateIdx, t)
					s := tok.score + selfTrans + acScore
					addOrRecombine(&nextTokens, nextPool, recom, -1, tok.stateIdx, tok.history, s)
					if s > bestNextScore {
						bestNextScore = s
					}
				}

				// Forward
				nextState := tok.stateIdx + 1
				if nextState <= acoustic.NumEmittingStates {
					fwdTrans := hmm.TransLog[tok.stateIdx][nextState]
					if fwdTrans > mathutil.LogZero+1 {
						acScore := cachedEmit(hmmOrd, nextState, t)
						s := tok.score + fwdTrans + acScore
						addOrRecombine(&nextTokens, nextPool, recom, -1, nextState, tok.history, s)
						if s > bestNextScore {
							bestNextScore = s
						}
					}
				}

				// Exit → silence completed, expand to root children with look-ahead
				exitState := acoustic.NumStatesPerPhoneme - 1
				exitTrans := hmm.TransLog[tok.stateIdx][exitState]
				if exitTrans <= mathutil.LogZero+1 {
					exitTrans = math.Log(0.5)
				}
				if exitTrans > mathutil.LogZero+1 {
					baseScore := tok.score + exitTrans
					if bestNextScore > math.Inf(-1) && baseScore < bestNextScore-cfg.BeamWidth {
						continue
					}

					for _, c := range rootChildren {
						nd := &nodes[c.nodeIdx]
						acScore := cachedEmit(nd.hmmOrd, 1, t)
						s := baseScore + acScore + lmLookAhead
						addOrRecombine(&nextTokens, nextPool, recom, c.nodeIdx, 1, tok.history, s)
						if s > bestNextScore {
							bestNextScore = s
						}
					}

					// Re-enter silence
					{
						acScore := cachedEmit(silHMMOrd, 1, t)
						s := baseScore + acScore
						addOrRecombine(&nextTokens, nextPool, recom, -1, 1, tok.history, s)
						if s > bestNextScore {
							bestNextScore = s
						}
					}
				}
				continue
			}

			// === Trie token ===
			nd := &nodes[tok.nodeIdx]
			hmm := nd.hmm
			hmmOrd := nd.hmmOrd

			// Self-loop
			selfTrans := hmm.TransLog[tok.stateIdx][tok.stateIdx]
			if selfTrans > mathutil.LogZero+1 {
				acScore := cachedEmit(hmmOrd, tok.stateIdx, t)
				s := tok.score + selfTrans + acScore
				addOrRecombine(&nextTokens, nextPool, recom, tok.nodeIdx, tok.stateIdx, tok.history, s)
				if s > bestNextScore {
					bestNextScore = s
				}
			}

			// Forward
			nextState := tok.stateIdx + 1
			if nextState <= acoustic.NumEmittingStates {
				fwdTrans := hmm.TransLog[tok.stateIdx][nextState]
				if fwdTrans > mathutil.LogZero+1 {
					acScore := cachedEmit(hmmOrd, nextState, t)
					s := tok.score + fwdTrans + acScore
					addOrRecombine(&nextTokens, nextPool, recom, tok.nodeIdx, nextState, tok.history, s)
					if s > bestNextScore {
						bestNextScore = s
					}
				}
			}

			// Exit: phoneme completed
			exitState := acoustic.NumStatesPerPhoneme - 1
			exitTrans := hmm.TransLog[tok.stateIdx][exitState]
			if exitTrans <= mathutil.LogZero+1 {
				exitTrans = math.Log(0.5)
			}
			if exitTrans > mathutil.LogZero+1 {
				baseScore := tok.score + exitTrans

				// Transition to child nodes (next phoneme within a word)
				for _, c := range nd.children {
					childNd := &nodes[c.nodeIdx]
					acScore := cachedEmit(childNd.hmmOrd, 1, t)
					s := baseScore + acScore
					addOrRecombine(&nextTokens, nextPool, recom, c.nodeIdx, 1, tok.history, s)
					if s > bestNextScore {
						bestNextScore = s
					}
				}

				// Word-end: this node completes one or more words
				if len(nd.wordEnds) > 0 {
					if bestNextScore > math.Inf(-1) && baseScore < bestNextScore-cfg.BeamWidth {
						continue
					}

					for _, word := range nd.wordEnds {
						newNode := &wordHistoryNode{
							word:  word,
							frame: t,
							prev:  tok.history,
						}
						if tok.history != nil {
							newNode.length = tok.history.length + 1
						} else {
							newNode.length = 1
						}

						// Compute actual LM score, subtract look-ahead that was applied at trie entry
						lmHistBuf = buildLMHist(tok.history, lmHistBuf[:0])
						var lmScore float64
						if len(lmHistBuf) == 0 {
							lmScore = scoreLM([]string{"<s>"}, word)
						} else {
							lmScore = scoreLM(lmHistBuf, word)
						}
						// baseScore includes lmLookAhead from trie entry; replace with actual LM
						wordBaseScore := baseScore - lmLookAhead + lmScore + cfg.WordInsertionPenalty

						// Start new words: expand root children with look-ahead
						for _, c := range rootChildren {
							childNd := &nodes[c.nodeIdx]
							acScore := cachedEmit(childNd.hmmOrd, 1, t)
							s := wordBaseScore + acScore + lmLookAhead
							addOrRecombine(&nextTokens, nextPool, recom, c.nodeIdx, 1, newNode, s)
							if s > bestNextScore {
								bestNextScore = s
							}
						}

						// Enter silence (no look-ahead for silence)
						if silHMM != nil {
							acScore := cachedEmit(silHMMOrd, 1, t)
							s := wordBaseScore + acScore
							addOrRecombine(&nextTokens, nextPool, recom, -1, 1, newNode, s)
							if s > bestNextScore {
								bestNextScore = s
							}
						}
					}
				}
			}
		}

		// Beam pruning
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

	// Build result: for trie tokens at word-end nodes, apply actual LM scoring
	var finalNode *wordHistoryNode
	if best.nodeIdx == -1 {
		finalNode = best.history
	} else {
		nd := &nodes[best.nodeIdx]
		if len(nd.wordEnds) > 0 {
			// Try each word-end with LM scoring
			lmHistBuf = buildLMHist(best.history, lmHistBuf[:0])
			bestWordScore := math.Inf(-1)
			bestWord := ""
			for _, w := range nd.wordEnds {
				var lmScore float64
				if len(lmHistBuf) == 0 {
					lmScore = scoreLM([]string{"<s>"}, w)
				} else {
					lmScore = scoreLM(lmHistBuf, w)
				}
				s := best.score - lmLookAhead + lmScore + cfg.WordInsertionPenalty
				if s > bestWordScore {
					bestWordScore = s
					bestWord = w
				}
			}
			if bestWord != "" && bestWordScore > best.score-cfg.BeamWidth {
				finalNode = &wordHistoryNode{
					word: bestWord,
					prev: best.history,
				}
				if best.history != nil {
					finalNode.length = best.history.length + 1
				} else {
					finalNode.length = 1
				}
			} else {
				finalNode = best.history
			}
		} else {
			finalNode = best.history
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

// addOrRecombine adds a token or replaces an existing one at the same (nodeIdx, stateIdx, lastWord).
func addOrRecombine(tokens *[]*token, pool *tokenPool, recom map[recomKey]int, nodeIdx, stateIdx int, history *wordHistoryNode, score float64) {
	lw := ""
	if history != nil {
		lw = history.word
	}
	key := recomKey{nodeIdx: nodeIdx, stateIdx: stateIdx, lastWord: lw}
	if idx, exists := recom[key]; exists {
		if score > (*tokens)[idx].score {
			(*tokens)[idx].score = score
			(*tokens)[idx].history = history
		}
		return
	}
	nt := pool.get()
	nt.score = score
	nt.nodeIdx = nodeIdx
	nt.stateIdx = stateIdx
	nt.history = history
	recom[key] = len(*tokens)
	*tokens = append(*tokens, nt)
}

// buildLMHist builds the LM history from a wordHistoryNode chain into the provided buffer.
func buildLMHist(n *wordHistoryNode, buf []string) []string {
	if n == nil {
		return buf
	}
	cur := n
	for cur != nil {
		buf = append(buf, cur.word)
		cur = cur.prev
	}
	for i, j := 0, len(buf)-1; i < j; i, j = i+1, j-1 {
		buf[i], buf[j] = buf[j], buf[i]
	}
	return buf
}

func pruneTokens(src []*token, dst []*token, beamWidth float64, maxActive int) []*token {
	if len(src) == 0 {
		return dst
	}

	bestScore := src[0].score
	for _, tok := range src[1:] {
		if tok.score > bestScore {
			bestScore = tok.score
		}
	}

	threshold := bestScore - beamWidth
	for _, tok := range src {
		if tok.score >= threshold {
			dst = append(dst, tok)
		}
	}

	if len(dst) > maxActive {
		sort.Slice(dst, func(i, j int) bool {
			return dst[i].score > dst[j].score
		})
		dst = dst[:maxActive]
	}

	return dst
}
