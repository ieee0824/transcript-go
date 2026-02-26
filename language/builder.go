package language

import (
	"fmt"
	"io"
	"math"
	"sort"
)

// Builder accumulates sentences and builds an N-gram language model.
type Builder struct {
	order    int
	unigrams map[string]int
	bigrams  map[[2]string]int
	trigrams map[[3]string]int
	fourgrams map[[4]string]int
	Log      io.Writer // if non-nil, progress is written here
}

// NewBuilder creates a new N-gram builder.
// order must be 2 (bigram), 3 (trigram), or 4 (fourgram).
func NewBuilder(order int) *Builder {
	if order < 2 {
		order = 2
	}
	if order > 4 {
		order = 4
	}
	return &Builder{
		order:     order,
		unigrams:  make(map[string]int),
		bigrams:   make(map[[2]string]int),
		trigrams:  make(map[[3]string]int),
		fourgrams: make(map[[4]string]int),
	}
}

// AddSentence adds a tokenized sentence. <s> and </s> are added automatically.
func (b *Builder) AddSentence(words []string) {
	if len(words) == 0 {
		return
	}
	// Prepend <s>, append </s>
	seq := make([]string, 0, len(words)+2)
	seq = append(seq, "<s>")
	seq = append(seq, words...)
	seq = append(seq, "</s>")

	for i := 0; i < len(seq); i++ {
		b.unigrams[seq[i]]++

		if i >= 1 {
			b.bigrams[[2]string{seq[i-1], seq[i]}]++
		}
		if b.order >= 3 && i >= 2 {
			b.trigrams[[3]string{seq[i-2], seq[i-1], seq[i]}]++
		}
		if b.order >= 4 && i >= 3 {
			b.fourgrams[[4]string{seq[i-3], seq[i-2], seq[i-1], seq[i]}]++
		}
	}
}

// logf writes a progress message to b.Log if set.
func (b *Builder) logf(format string, args ...interface{}) {
	if b.Log != nil {
		fmt.Fprintf(b.Log, format, args...)
	}
}

// WriteARPA writes the model in ARPA format (log10 probabilities) to w.
// Uses Witten-Bell smoothing.
func (b *Builder) WriteARPA(w io.Writer) error {
	b.logf("  [WriteARPA] stats: %d unigrams, %d bigrams, %d trigrams, %d fourgrams\n",
		len(b.unigrams), len(b.bigrams), len(b.trigrams), len(b.fourgrams))
	b.logf("  [WriteARPA] computing context counts...\n")
	// Compute total unigram count (excluding <s> which is history-only)
	uniTotal := 0
	for _, c := range b.unigrams {
		uniTotal += c
	}

	// Witten-Bell: for each history h, T(h) = number of unique words following h
	// bigram contexts
	biContextTotal := make(map[string]int)    // h -> N(h)
	biContextTypes := make(map[string]int)    // h -> T(h)
	for key, c := range b.bigrams {
		biContextTotal[key[0]] += c
		_ = c
		biContextTypes[key[0]]++
	}

	// trigram contexts
	triContextTotal := make(map[[2]string]int) // (h1,h2) -> N(h1,h2)
	triContextTypes := make(map[[2]string]int) // (h1,h2) -> T(h1,h2)
	for key, c := range b.trigrams {
		ctx := [2]string{key[0], key[1]}
		triContextTotal[ctx] += c
		_ = c
		triContextTypes[ctx]++
	}

	// fourgram contexts
	fourContextTotal := make(map[[3]string]int) // (h1,h2,h3) -> N(h1,h2,h3)
	fourContextTypes := make(map[[3]string]int) // (h1,h2,h3) -> T(h1,h2,h3)
	for key, c := range b.fourgrams {
		ctx := [3]string{key[0], key[1], key[2]}
		fourContextTotal[ctx] += c
		_ = c
		fourContextTypes[ctx]++
	}

	b.logf("  [WriteARPA] pre-indexing n-grams by context...\n")
	// Pre-index n-grams by context for O(1) lookup instead of linear scan.
	type bigramEntry struct {
		word  string
		count int
	}
	bigramsByCtx := make(map[string][]bigramEntry, len(biContextTotal))
	for key, c := range b.bigrams {
		bigramsByCtx[key[0]] = append(bigramsByCtx[key[0]], bigramEntry{key[1], c})
	}

	type trigramEntry struct {
		word  string
		count int
	}
	trigramsByCtx := make(map[[2]string][]trigramEntry, len(triContextTotal))
	for key, c := range b.trigrams {
		ctx := [2]string{key[0], key[1]}
		trigramsByCtx[ctx] = append(trigramsByCtx[ctx], trigramEntry{key[2], c})
	}

	type fourgramEntry struct {
		word  string
		count int
	}
	fourgramsByCtx := make(map[[3]string][]fourgramEntry, len(fourContextTotal))
	for key, c := range b.fourgrams {
		ctx := [3]string{key[0], key[1], key[2]}
		fourgramsByCtx[ctx] = append(fourgramsByCtx[ctx], fourgramEntry{key[3], c})
	}

	b.logf("  [WriteARPA] computing unigram probabilities (%d)...\n", len(b.unigrams))
	// --- Compute unigram probabilities ---
	// P(w) = C(w) / N  (MLE, will be adjusted by backoff)
	type uniProb struct {
		word       string
		logProb    float64 // log10
		logBackoff float64 // log10
	}
	unis := make([]uniProb, 0, len(b.unigrams))
	for word, count := range b.unigrams {
		lp := math.Log10(float64(count) / float64(uniTotal))

		// Backoff weight for unigram -> used when bigram backs off to unigram
		// bow(w) = (1 - sum_of_discounted_bigram_probs_with_h=w) / (1 - sum_of_unigram_probs_of_seen_bigram_words)
		// With Witten-Bell: bow = T(h) / (N(h) + T(h))  as the leftover mass ratio
		var bo float64
		if n, ok := biContextTotal[word]; ok {
			t := biContextTypes[word]
			// Sum of Witten-Bell bigram probs for observed bigrams
			sumBiProb := 0.0
			for _, e := range bigramsByCtx[word] {
				sumBiProb += float64(e.count) / float64(n+t)
			}
			// Sum of unigram probs for words seen in bigram with this context
			sumUniProb := 0.0
			for _, e := range bigramsByCtx[word] {
				sumUniProb += float64(b.unigrams[e.word]) / float64(uniTotal)
			}
			if sumUniProb < 1.0 {
				bo = math.Log10((1.0 - sumBiProb) / (1.0 - sumUniProb))
			}
		}

		unis = append(unis, uniProb{word, lp, bo})
	}
	sort.Slice(unis, func(i, j int) bool { return unis[i].word < unis[j].word })

	b.logf("  [WriteARPA] computing bigram probabilities (%d)...\n", len(b.bigrams))
	// --- Compute bigram probabilities (Witten-Bell) ---
	type biProb struct {
		key        [2]string
		logProb    float64
		logBackoff float64
	}
	bis := make([]biProb, 0, len(b.bigrams))
	for key, count := range b.bigrams {
		h := key[0]
		n := biContextTotal[h]
		t := biContextTypes[h]
		// P_wb(w|h) = C(h,w) / (N(h) + T(h))
		lp := math.Log10(float64(count) / float64(n+t))

		// Backoff weight for bigram -> trigram backoff
		var bo float64
		if b.order >= 3 {
			ctx := [2]string{key[0], key[1]}
			if tn, ok := triContextTotal[ctx]; ok {
				tt := triContextTypes[ctx]
				sumTriProb := 0.0
				for _, e := range trigramsByCtx[ctx] {
					sumTriProb += float64(e.count) / float64(tn+tt)
				}
				sumBiProb := 0.0
				for _, e := range trigramsByCtx[ctx] {
					w := e.word
					biKey := [2]string{ctx[1], w}
					if bc, ok := b.bigrams[biKey]; ok {
						bn := biContextTotal[ctx[1]]
						bt := biContextTypes[ctx[1]]
						sumBiProb += float64(bc) / float64(bn+bt)
					} else {
						sumBiProb += float64(b.unigrams[w]) / float64(uniTotal)
					}
				}
				if sumBiProb < 1.0 {
					bo = math.Log10((1.0 - sumTriProb) / (1.0 - sumBiProb))
				}
			}
		}

		bis = append(bis, biProb{key, lp, bo})
	}
	sort.Slice(bis, func(i, j int) bool {
		if bis[i].key[0] != bis[j].key[0] {
			return bis[i].key[0] < bis[j].key[0]
		}
		return bis[i].key[1] < bis[j].key[1]
	})

	b.logf("  [WriteARPA] computing trigram probabilities (%d)...\n", len(b.trigrams))
	// --- Compute trigram probabilities (Witten-Bell) ---
	type triProb struct {
		key        [3]string
		logProb    float64
		logBackoff float64
	}
	tris := make([]triProb, 0, len(b.trigrams))
	if b.order >= 3 {
		for key, count := range b.trigrams {
			ctx := [2]string{key[0], key[1]}
			n := triContextTotal[ctx]
			t := triContextTypes[ctx]
			lp := math.Log10(float64(count) / float64(n+t))

			// Backoff weight for trigram -> fourgram backoff
			var bo float64
			if b.order >= 4 {
				fctx := [3]string{key[0], key[1], key[2]}
				if fn, ok := fourContextTotal[fctx]; ok {
					ft := fourContextTypes[fctx]
					sumFourProb := 0.0
					for _, e := range fourgramsByCtx[fctx] {
						sumFourProb += float64(e.count) / float64(fn+ft)
					}
					sumTriProb := 0.0
					for _, e := range fourgramsByCtx[fctx] {
						w := e.word
						triKey := [2]string{fctx[1], fctx[2]}
						tn2 := triContextTotal[triKey]
						tt2 := triContextTypes[triKey]
						if tc, ok := b.trigrams[[3]string{fctx[1], fctx[2], w}]; ok {
							sumTriProb += float64(tc) / float64(tn2+tt2)
						} else {
							biKey := [2]string{fctx[2], w}
							if bc, ok := b.bigrams[biKey]; ok {
								bn := biContextTotal[fctx[2]]
								bt := biContextTypes[fctx[2]]
								sumTriProb += float64(bc) / float64(bn+bt)
							} else {
								sumTriProb += float64(b.unigrams[w]) / float64(uniTotal)
							}
						}
					}
					if sumTriProb < 1.0 {
						bo = math.Log10((1.0 - sumFourProb) / (1.0 - sumTriProb))
					}
				}
			}

			tris = append(tris, triProb{key, lp, bo})
		}
		sort.Slice(tris, func(i, j int) bool {
			if tris[i].key[0] != tris[j].key[0] {
				return tris[i].key[0] < tris[j].key[0]
			}
			if tris[i].key[1] != tris[j].key[1] {
				return tris[i].key[1] < tris[j].key[1]
			}
			return tris[i].key[2] < tris[j].key[2]
		})
	}

	b.logf("  [WriteARPA] computing fourgram probabilities (%d)...\n", len(b.fourgrams))
	// --- Compute fourgram probabilities (Witten-Bell) ---
	type fourProb struct {
		key     [4]string
		logProb float64
	}
	fours := make([]fourProb, 0, len(b.fourgrams))
	if b.order >= 4 {
		for key, count := range b.fourgrams {
			ctx := [3]string{key[0], key[1], key[2]}
			n := fourContextTotal[ctx]
			t := fourContextTypes[ctx]
			lp := math.Log10(float64(count) / float64(n+t))
			fours = append(fours, fourProb{key, lp})
		}
		sort.Slice(fours, func(i, j int) bool {
			for k := 0; k < 4; k++ {
				if fours[i].key[k] != fours[j].key[k] {
					return fours[i].key[k] < fours[j].key[k]
				}
			}
			return false
		})
	}

	b.logf("  [WriteARPA] writing ARPA output...\n")
	// --- Write ARPA ---
	fmt.Fprintln(w, "\\data\\")
	fmt.Fprintf(w, "ngram 1=%d\n", len(unis))
	fmt.Fprintf(w, "ngram 2=%d\n", len(bis))
	if b.order >= 3 && len(tris) > 0 {
		fmt.Fprintf(w, "ngram 3=%d\n", len(tris))
	}
	if b.order >= 4 && len(fours) > 0 {
		fmt.Fprintf(w, "ngram 4=%d\n", len(fours))
	}
	fmt.Fprintln(w)

	fmt.Fprintln(w, "\\1-grams:")
	for _, u := range unis {
		if u.logBackoff != 0 {
			fmt.Fprintf(w, "%.6f\t%s\t%.6f\n", u.logProb, u.word, u.logBackoff)
		} else {
			fmt.Fprintf(w, "%.6f\t%s\n", u.logProb, u.word)
		}
	}
	fmt.Fprintln(w)

	fmt.Fprintln(w, "\\2-grams:")
	for _, bi := range bis {
		if (b.order >= 3) && bi.logBackoff != 0 {
			fmt.Fprintf(w, "%.6f\t%s %s\t%.6f\n", bi.logProb, bi.key[0], bi.key[1], bi.logBackoff)
		} else {
			fmt.Fprintf(w, "%.6f\t%s %s\n", bi.logProb, bi.key[0], bi.key[1])
		}
	}
	fmt.Fprintln(w)

	if b.order >= 3 && len(tris) > 0 {
		fmt.Fprintln(w, "\\3-grams:")
		for _, tri := range tris {
			if b.order >= 4 && tri.logBackoff != 0 {
				fmt.Fprintf(w, "%.6f\t%s %s %s\t%.6f\n", tri.logProb, tri.key[0], tri.key[1], tri.key[2], tri.logBackoff)
			} else {
				fmt.Fprintf(w, "%.6f\t%s %s %s\n", tri.logProb, tri.key[0], tri.key[1], tri.key[2])
			}
		}
		fmt.Fprintln(w)
	}

	if b.order >= 4 && len(fours) > 0 {
		fmt.Fprintln(w, "\\4-grams:")
		for _, f := range fours {
			fmt.Fprintf(w, "%.6f\t%s %s %s %s\n", f.logProb, f.key[0], f.key[1], f.key[2], f.key[3])
		}
		fmt.Fprintln(w)
	}

	fmt.Fprintln(w, "\\end\\")
	return nil
}
