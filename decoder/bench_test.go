package decoder

import (
	"math"
	"math/rand"
	"strings"
	"testing"
	"github.com/ieee0824/transcript-go/acoustic"
	"github.com/ieee0824/transcript-go/language"
	"github.com/ieee0824/transcript-go/lexicon"
)

func buildBenchModel(vocabSize int) (*acoustic.AcousticModel, *language.NGramModel, *lexicon.Dictionary) {
	dim := 39
	numMix := 2

	am := &acoustic.AcousticModel{
		Phonemes:   make(map[acoustic.Phoneme]*acoustic.PhonemeHMM),
		FeatureDim: dim,
		NumMix:     numMix,
	}
	for _, p := range acoustic.AllPhonemes() {
		am.Phonemes[p] = acoustic.NewPhonemeHMM(p, dim, numMix)
	}

	// Build a small vocabulary
	phonemes := []acoustic.Phoneme{acoustic.PhonA, acoustic.PhonI, acoustic.PhonU, acoustic.PhonE, acoustic.PhonO}
	words := []string{"あ", "い", "う", "え", "お", "か", "き", "く", "け", "こ",
		"さ", "し", "す", "せ", "そ", "た", "ち", "つ", "て", "と"}

	if vocabSize > len(words) {
		vocabSize = len(words)
	}
	words = words[:vocabSize]

	// Build ARPA
	var sb strings.Builder
	sb.WriteString("\\data\\\n")
	sb.WriteString("ngram 1=" + strings.Repeat("0", 0))
	uniCount := len(words) + 2 // +<s>, </s>
	biCount := len(words)
	sb.Reset()
	sb.WriteString("\\data\\\n")
	sb.WriteString("ngram 1=" + itoa(uniCount) + "\n")
	sb.WriteString("ngram 2=" + itoa(biCount) + "\n\n")
	sb.WriteString("\\1-grams:\n")
	sb.WriteString("-1.0\t</s>\n")
	sb.WriteString("-1.0\t<s>\t0.0\n")
	lp := math.Log10(1.0 / float64(len(words)))
	for _, w := range words {
		sb.WriteString(ftoa(lp) + "\t" + w + "\t0.0\n")
	}
	sb.WriteString("\n\\2-grams:\n")
	for _, w := range words {
		sb.WriteString(ftoa(lp) + "\t<s>\t" + w + "\n")
	}
	sb.WriteString("\n\\end\\\n")

	lm, _ := language.LoadARPA(strings.NewReader(sb.String()))

	dict := lexicon.NewDictionary()
	for i, w := range words {
		ph := phonemes[i%len(phonemes)]
		dict.Add(w, w, []acoustic.Phoneme{ph})
	}

	return am, lm, dict
}

func itoa(n int) string {
	return strings.TrimRight(strings.TrimRight(
		strings.Replace(
			strings.Replace(
				strings.Replace(
					strings.Replace(
						strings.Replace("00000"+
							string(rune('0'+n%10)),
							"00000", "", 1),
						"0000", "", 1),
					"000", "", 1),
				"00", "", 1),
			"0", "", 1),
		"0"), "")
}

func ftoa(f float64) string {
	s := ""
	if f < 0 {
		s = "-"
		f = -f
	}
	whole := int(f)
	frac := int((f - float64(whole)) * 10000)
	ws := string(rune('0' + whole))
	fs := string([]rune{rune('0' + frac/1000), rune('0' + (frac/100)%10), rune('0' + (frac/10)%10), rune('0' + frac%10)})
	return s + ws + "." + fs
}

func benchFeatures(T, dim int) [][]float64 {
	features := make([][]float64, T)
	for t := range features {
		features[t] = make([]float64, dim)
		for d := range features[t] {
			features[t][d] = rand.NormFloat64()
		}
	}
	return features
}

func BenchmarkDecode_5vocab_50frames(b *testing.B) {
	am, lm, dict := buildBenchModel(5)
	features := benchFeatures(50, 39)
	cfg := Config{BeamWidth: 200, MaxActiveTokens: 500, LMWeight: 1.0, WordInsertionPenalty: -2.0}
	b.ResetTimer()
	for b.Loop() {
		Decode(features, am, lm, dict, cfg)
	}
}

func BenchmarkDecode_10vocab_100frames(b *testing.B) {
	am, lm, dict := buildBenchModel(10)
	features := benchFeatures(100, 39)
	cfg := Config{BeamWidth: 200, MaxActiveTokens: 500, LMWeight: 1.0, WordInsertionPenalty: -2.0}
	b.ResetTimer()
	for b.Loop() {
		Decode(features, am, lm, dict, cfg)
	}
}

func BenchmarkDecode_20vocab_200frames(b *testing.B) {
	am, lm, dict := buildBenchModel(20)
	features := benchFeatures(200, 39)
	cfg := Config{BeamWidth: 200, MaxActiveTokens: 500, LMWeight: 1.0, WordInsertionPenalty: -2.0}
	b.ResetTimer()
	for b.Loop() {
		Decode(features, am, lm, dict, cfg)
	}
}
