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
}

// NewBuilder creates a new N-gram builder.
// order must be 2 (bigram) or 3 (trigram).
func NewBuilder(order int) *Builder {
	if order < 2 {
		order = 2
	}
	if order > 3 {
		order = 3
	}
	return &Builder{
		order:    order,
		unigrams: make(map[string]int),
		bigrams:  make(map[[2]string]int),
		trigrams: make(map[[3]string]int),
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
	}
}

// WriteARPA writes the model in ARPA format (log10 probabilities) to w.
// Uses Witten-Bell smoothing.
func (b *Builder) WriteARPA(w io.Writer) error {
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
			for key, c := range b.bigrams {
				if key[0] == word {
					sumBiProb += float64(c) / float64(n+t)
				}
			}
			// Sum of unigram probs for words seen in bigram with this context
			sumUniProb := 0.0
			for key := range b.bigrams {
				if key[0] == word {
					sumUniProb += float64(b.unigrams[key[1]]) / float64(uniTotal)
				}
			}
			if sumUniProb < 1.0 {
				bo = math.Log10((1.0 - sumBiProb) / (1.0 - sumUniProb))
			}
		}

		unis = append(unis, uniProb{word, lp, bo})
	}
	sort.Slice(unis, func(i, j int) bool { return unis[i].word < unis[j].word })

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
				for tkey, tc := range b.trigrams {
					if tkey[0] == ctx[0] && tkey[1] == ctx[1] {
						sumTriProb += float64(tc) / float64(tn+tt)
					}
				}
				sumBiProb := 0.0
				for tkey := range b.trigrams {
					if tkey[0] == ctx[0] && tkey[1] == ctx[1] {
						w := tkey[2]
						biKey := [2]string{ctx[1], w}
						if bc, ok := b.bigrams[biKey]; ok {
							bn := biContextTotal[ctx[1]]
							bt := biContextTypes[ctx[1]]
							sumBiProb += float64(bc) / float64(bn+bt)
						} else {
							sumBiProb += float64(b.unigrams[w]) / float64(uniTotal)
						}
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

	// --- Compute trigram probabilities (Witten-Bell) ---
	type triProb struct {
		key     [3]string
		logProb float64
	}
	tris := make([]triProb, 0, len(b.trigrams))
	if b.order >= 3 {
		for key, count := range b.trigrams {
			ctx := [2]string{key[0], key[1]}
			n := triContextTotal[ctx]
			t := triContextTypes[ctx]
			lp := math.Log10(float64(count) / float64(n+t))
			tris = append(tris, triProb{key, lp})
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

	// --- Write ARPA ---
	fmt.Fprintln(w, "\\data\\")
	fmt.Fprintf(w, "ngram 1=%d\n", len(unis))
	fmt.Fprintf(w, "ngram 2=%d\n", len(bis))
	if b.order >= 3 && len(tris) > 0 {
		fmt.Fprintf(w, "ngram 3=%d\n", len(tris))
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
		if b.order >= 3 && bi.logBackoff != 0 {
			fmt.Fprintf(w, "%.6f\t%s %s\t%.6f\n", bi.logProb, bi.key[0], bi.key[1], bi.logBackoff)
		} else {
			fmt.Fprintf(w, "%.6f\t%s %s\n", bi.logProb, bi.key[0], bi.key[1])
		}
	}
	fmt.Fprintln(w)

	if b.order >= 3 && len(tris) > 0 {
		fmt.Fprintln(w, "\\3-grams:")
		for _, tri := range tris {
			fmt.Fprintf(w, "%.6f\t%s %s %s\n", tri.logProb, tri.key[0], tri.key[1], tri.key[2])
		}
		fmt.Fprintln(w)
	}

	fmt.Fprintln(w, "\\end\\")
	return nil
}
