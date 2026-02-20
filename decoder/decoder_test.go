package decoder

import (
	"math"
	"strings"
	"testing"
	"github.com/ieee0824/transcript-go/acoustic"
	"github.com/ieee0824/transcript-go/language"
	"github.com/ieee0824/transcript-go/lexicon"
)

// buildTinyModel creates a minimal model for testing:
// Vocabulary: "あ" (phonemes: [a]), "い" (phonemes: [i])
// Each phoneme has a 1D GMM with distinct means.
func buildTinyModel() (*acoustic.AcousticModel, *language.NGramModel, *lexicon.Dictionary) {
	dim := 1
	numMix := 1

	am := &acoustic.AcousticModel{
		Phonemes:   make(map[acoustic.Phoneme]*acoustic.PhonemeHMM),
		FeatureDim: dim,
		NumMix:     numMix,
	}

	// Phoneme "a" -> GMM mean=0.0
	am.Phonemes[acoustic.PhonA] = acoustic.NewPhonemeHMM(acoustic.PhonA, dim, numMix)
	setHMMGMM(am.Phonemes[acoustic.PhonA], 0.0)

	// Phoneme "i" -> GMM mean=5.0
	am.Phonemes[acoustic.PhonI] = acoustic.NewPhonemeHMM(acoustic.PhonI, dim, numMix)
	setHMMGMM(am.Phonemes[acoustic.PhonI], 5.0)

	// Language model
	arpa := `\data\
ngram 1=4
ngram 2=4

\1-grams:
-1.0	</s>
-1.0	<s>	0.0
-0.5	あ	0.0
-0.5	い	0.0

\2-grams:
-0.3	<s>	あ
-0.3	<s>	い
-0.3	あ	い
-0.3	い	あ

\end\
`
	lm, _ := language.LoadARPA(strings.NewReader(arpa))

	// Dictionary
	dict := lexicon.NewDictionary()
	dict.Add("あ", "ア", []acoustic.Phoneme{acoustic.PhonA})
	dict.Add("い", "イ", []acoustic.Phoneme{acoustic.PhonI})

	return am, lm, dict
}

func setHMMGMM(hmm *acoustic.PhonemeHMM, mean float64) {
	for i := 1; i <= acoustic.NumEmittingStates; i++ {
		hmm.States[i].GMM = acoustic.NewGMMWithParams(
			[][]float64{{mean}},
			[][]float64{{0.5}},
			[]float64{0.0}, // log(1.0)
		)
	}
}

func TestDecode_SingleWord(t *testing.T) {
	am, lm, dict := buildTinyModel()
	cfg := Config{
		BeamWidth:            300.0,
		MaxActiveTokens:      500,
		LMWeight:             1.0,
		WordInsertionPenalty: -5.0,
	}

	// Feature frames close to mean=0.0 -> should recognize "あ"
	features := make([][]float64, 6)
	for i := range features {
		features[i] = []float64{0.1}
	}

	result := Decode(features, am, lm, dict, cfg)
	if result == nil {
		t.Fatal("result is nil")
	}
	if result.Text == "" {
		t.Fatal("result text is empty")
	}

	// The text should contain "あ" and not "い"
	if !strings.Contains(result.Text, "あ") {
		t.Errorf("expected text to contain あ, got %q", result.Text)
	}
}

func TestDecode_TwoWords(t *testing.T) {
	am, lm, dict := buildTinyModel()
	cfg := Config{
		BeamWidth:            300.0,
		MaxActiveTokens:      500,
		LMWeight:             1.0,
		WordInsertionPenalty: -2.0,
	}

	// First half near 0 ("あ"), second half near 5 ("い")
	features := make([][]float64, 12)
	for i := 0; i < 6; i++ {
		features[i] = []float64{0.1}
	}
	for i := 6; i < 12; i++ {
		features[i] = []float64{4.9}
	}

	result := Decode(features, am, lm, dict, cfg)
	if result == nil {
		t.Fatal("result is nil")
	}

	// Should contain both あ and い
	hasA := strings.Contains(result.Text, "あ")
	hasI := strings.Contains(result.Text, "い")
	if !hasA || !hasI {
		t.Errorf("expected text with あ and い, got %q", result.Text)
	}
}

func TestDecode_EmptyFeatures(t *testing.T) {
	am, lm, dict := buildTinyModel()
	cfg := DefaultConfig()

	result := Decode(nil, am, lm, dict, cfg)
	if result.Text != "" {
		t.Errorf("expected empty text for nil features, got %q", result.Text)
	}
}

func TestDecode_ScoreFinite(t *testing.T) {
	am, lm, dict := buildTinyModel()
	cfg := Config{
		BeamWidth:            300.0,
		MaxActiveTokens:      500,
		LMWeight:             1.0,
		WordInsertionPenalty: 0.0,
	}

	features := make([][]float64, 6)
	for i := range features {
		features[i] = []float64{0.0}
	}

	result := Decode(features, am, lm, dict, cfg)
	if math.IsNaN(result.LogScore) || math.IsInf(result.LogScore, 0) {
		t.Errorf("LogScore = %f (not finite)", result.LogScore)
	}
}

// TestDecode_MaxWordEndsZero verifies that MaxWordEnds=0 (unlimited) produces same result as legacy.
func TestDecode_MaxWordEndsZero(t *testing.T) {
	am, lm, dict := buildTinyModel()
	cfg := Config{
		BeamWidth:            300.0,
		MaxActiveTokens:      500,
		LMWeight:             1.0,
		WordInsertionPenalty: -2.0,
		MaxWordEnds:          0,
	}

	features := make([][]float64, 12)
	for i := 0; i < 6; i++ {
		features[i] = []float64{0.1}
	}
	for i := 6; i < 12; i++ {
		features[i] = []float64{4.9}
	}

	result := Decode(features, am, lm, dict, cfg)
	if result == nil {
		t.Fatal("result is nil")
	}
	hasA := strings.Contains(result.Text, "あ")
	hasI := strings.Contains(result.Text, "い")
	if !hasA || !hasI {
		t.Errorf("expected あ and い, got %q", result.Text)
	}
}

// TestDecode_MaxWordEndsLimiting verifies pruning doesn't crash and produces output.
func TestDecode_MaxWordEndsLimiting(t *testing.T) {
	am, lm, dict := buildTinyModel()
	features := make([][]float64, 12)
	for i := 0; i < 6; i++ {
		features[i] = []float64{0.1}
	}
	for i := 6; i < 12; i++ {
		features[i] = []float64{4.9}
	}

	cfg1 := Config{BeamWidth: 300, MaxActiveTokens: 500, LMWeight: 1.0, MaxWordEnds: 1}
	cfg2 := Config{BeamWidth: 300, MaxActiveTokens: 500, LMWeight: 1.0, MaxWordEnds: 0}

	r1 := Decode(features, am, lm, dict, cfg1)
	r2 := Decode(features, am, lm, dict, cfg2)

	if r1.Text == "" {
		t.Error("MaxWordEnds=1 produced empty text")
	}
	if math.IsNaN(r1.LogScore) || math.IsInf(r1.LogScore, 0) {
		t.Errorf("MaxWordEnds=1 score not finite: %f", r1.LogScore)
	}
	// Unlimited should produce equal or better score
	if r2.LogScore < r1.LogScore-1.0 {
		t.Errorf("unlimited score %.4f significantly worse than limited %.4f", r2.LogScore, r1.LogScore)
	}
}

// buildTinyTrigramModel creates a model with a trigram LM for testing trigram recombination.
func buildTinyTrigramModel() (*acoustic.AcousticModel, *language.NGramModel, *lexicon.Dictionary) {
	dim := 1
	numMix := 1

	am := &acoustic.AcousticModel{
		Phonemes:   make(map[acoustic.Phoneme]*acoustic.PhonemeHMM),
		FeatureDim: dim,
		NumMix:     numMix,
	}
	am.Phonemes[acoustic.PhonA] = acoustic.NewPhonemeHMM(acoustic.PhonA, dim, numMix)
	setHMMGMM(am.Phonemes[acoustic.PhonA], 0.0)
	am.Phonemes[acoustic.PhonI] = acoustic.NewPhonemeHMM(acoustic.PhonI, dim, numMix)
	setHMMGMM(am.Phonemes[acoustic.PhonI], 5.0)

	arpa := `\data\
ngram 1=4
ngram 2=4
ngram 3=2

\1-grams:
-1.0	</s>
-1.0	<s>	0.0
-0.5	あ	0.0
-0.5	い	0.0

\2-grams:
-0.3	<s>	あ	0.0
-0.3	<s>	い	0.0
-0.3	あ	い	0.0
-0.3	い	あ	0.0

\3-grams:
-0.1	<s>	あ	い
-0.8	<s>	い	あ

\end\
`
	lm, _ := language.LoadARPA(strings.NewReader(arpa))

	dict := lexicon.NewDictionary()
	dict.Add("あ", "ア", []acoustic.Phoneme{acoustic.PhonA})
	dict.Add("い", "イ", []acoustic.Phoneme{acoustic.PhonI})
	return am, lm, dict
}

// TestDecode_Trigram verifies decoder works with a trigram LM.
func TestDecode_Trigram(t *testing.T) {
	am, lm, dict := buildTinyTrigramModel()
	cfg := Config{
		BeamWidth:            300.0,
		MaxActiveTokens:      500,
		LMWeight:             5.0,
		WordInsertionPenalty: -2.0,
	}

	// First half near 0 ("あ"), second half near 5 ("い")
	features := make([][]float64, 12)
	for i := 0; i < 6; i++ {
		features[i] = []float64{0.1}
	}
	for i := 6; i < 12; i++ {
		features[i] = []float64{4.9}
	}

	result := Decode(features, am, lm, dict, cfg)
	if result == nil {
		t.Fatal("result is nil")
	}

	hasA := strings.Contains(result.Text, "あ")
	hasI := strings.Contains(result.Text, "い")
	if !hasA || !hasI {
		t.Errorf("expected text with あ and い, got %q", result.Text)
	}
	if math.IsNaN(result.LogScore) || math.IsInf(result.LogScore, 0) {
		t.Errorf("LogScore = %f (not finite)", result.LogScore)
	}
}
