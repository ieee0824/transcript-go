package language

import (
	"bytes"
	"math"
	"strings"
	"testing"
)

func TestBuilderBigram(t *testing.T) {
	b := NewBuilder(2)
	b.AddSentence([]string{"東京", "タワー"})
	b.AddSentence([]string{"東京", "タワー", "に", "行く"})
	b.AddSentence([]string{"東京", "駅"})

	var buf bytes.Buffer
	if err := b.WriteARPA(&buf); err != nil {
		t.Fatalf("WriteARPA error: %v", err)
	}

	arpa := buf.String()
	t.Logf("ARPA output:\n%s", arpa)

	// Verify ARPA structure
	if !strings.Contains(arpa, "\\data\\") {
		t.Error("missing \\data\\ section")
	}
	if !strings.Contains(arpa, "\\1-grams:") {
		t.Error("missing \\1-grams: section")
	}
	if !strings.Contains(arpa, "\\2-grams:") {
		t.Error("missing \\2-grams: section")
	}
	if !strings.Contains(arpa, "\\end\\") {
		t.Error("missing \\end\\ section")
	}
	// Should not have trigrams for order=2
	if strings.Contains(arpa, "\\3-grams:") {
		t.Error("unexpected \\3-grams: section for bigram model")
	}

	// Verify it can be loaded back
	model, err := LoadARPA(strings.NewReader(arpa))
	if err != nil {
		t.Fatalf("LoadARPA error: %v", err)
	}
	if model.Order != 2 {
		t.Errorf("Order = %d, want 2", model.Order)
	}

	// Check that "東京" appears in vocabulary
	found := false
	for _, w := range model.Vocab() {
		if w == "東京" {
			found = true
			break
		}
	}
	if !found {
		t.Error("東京 not in vocabulary")
	}

	// Score a sentence — should be finite
	score := model.SentenceLogProb([]string{"東京", "タワー"})
	if math.IsNaN(score) || math.IsInf(score, 0) {
		t.Errorf("SentenceLogProb = %f (not finite)", score)
	}
	t.Logf("P(東京 タワー) = %.4f", score)
}

func TestBuilderTrigram(t *testing.T) {
	b := NewBuilder(3)
	b.AddSentence([]string{"今日", "は", "いい", "天気", "です"})
	b.AddSentence([]string{"今日", "は", "暑い", "です"})
	b.AddSentence([]string{"明日", "は", "いい", "天気", "です"})

	var buf bytes.Buffer
	if err := b.WriteARPA(&buf); err != nil {
		t.Fatalf("WriteARPA error: %v", err)
	}

	arpa := buf.String()
	t.Logf("ARPA output:\n%s", arpa)

	if !strings.Contains(arpa, "\\3-grams:") {
		t.Error("missing \\3-grams: section")
	}

	model, err := LoadARPA(strings.NewReader(arpa))
	if err != nil {
		t.Fatalf("LoadARPA error: %v", err)
	}
	if model.Order != 3 {
		t.Errorf("Order = %d, want 3", model.Order)
	}

	// P("今日 は いい 天気 です") should be higher than P("今日 は 寒い 天気 です")
	s1 := model.SentenceLogProb([]string{"今日", "は", "いい", "天気", "です"})
	s2 := model.SentenceLogProb([]string{"今日", "は", "寒い", "天気", "です"})
	t.Logf("P(今日はいい天気です) = %.4f", s1)
	t.Logf("P(今日は寒い天気です) = %.4f", s2)
	if s1 <= s2 {
		t.Errorf("seen sentence should score higher: %.4f <= %.4f", s1, s2)
	}
}

func TestBuilderRoundTrip(t *testing.T) {
	b := NewBuilder(2)
	sentences := [][]string{
		{"あ", "い"},
		{"あ", "い", "う"},
		{"い", "う"},
	}
	for _, s := range sentences {
		b.AddSentence(s)
	}

	var buf bytes.Buffer
	if err := b.WriteARPA(&buf); err != nil {
		t.Fatalf("WriteARPA error: %v", err)
	}

	model, err := LoadARPA(strings.NewReader(buf.String()))
	if err != nil {
		t.Fatalf("LoadARPA round-trip error: %v", err)
	}

	// All probabilities should be negative (log domain) and finite
	for w, e := range model.Unigrams {
		if e.LogProb >= 0 {
			t.Errorf("unigram %q LogProb = %f, want negative", w, e.LogProb)
		}
		if math.IsNaN(e.LogProb) || math.IsInf(e.LogProb, 0) {
			t.Errorf("unigram %q LogProb is not finite", w)
		}
	}
	for key, e := range model.Bigrams {
		if e.LogProb >= 0 {
			t.Errorf("bigram %v LogProb = %f, want negative", key, e.LogProb)
		}
	}
}
