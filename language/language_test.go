package language

import (
	"math"
	"strings"
	"testing"
)

const testARPA = `\data\
ngram 1=4
ngram 2=3

\1-grams:
-1.0	</s>
-1.0	<s>	-0.5
-0.5	東京
-0.7	タワー	-0.3

\2-grams:
-0.3	<s>	東京
-0.4	東京	タワー
-0.2	タワー	</s>

\end\
`

func TestLoadARPA(t *testing.T) {
	model, err := LoadARPA(strings.NewReader(testARPA))
	if err != nil {
		t.Fatalf("LoadARPA error: %v", err)
	}

	if model.Order != 2 {
		t.Errorf("Order = %d, want 2", model.Order)
	}
	if len(model.Unigrams) != 4 {
		t.Errorf("len(Unigrams) = %d, want 4", len(model.Unigrams))
	}
	if len(model.Bigrams) != 3 {
		t.Errorf("len(Bigrams) = %d, want 3", len(model.Bigrams))
	}

	// Check unigram for 東京: log10 prob = -0.5 -> ln prob = -0.5 * ln(10)
	if e, ok := model.Unigrams["東京"]; ok {
		want := -0.5 * math.Ln10
		if math.Abs(e.LogProb-want) > 1e-10 {
			t.Errorf("東京 unigram LogProb = %f, want %f", e.LogProb, want)
		}
	} else {
		t.Error("missing unigram for 東京")
	}
}

func TestLogProb_Bigram(t *testing.T) {
	model, err := LoadARPA(strings.NewReader(testARPA))
	if err != nil {
		t.Fatalf("LoadARPA error: %v", err)
	}

	// P(東京 | <s>) should use the bigram
	lp := model.LogProb([]string{"<s>"}, "東京")
	want := -0.3 * math.Ln10
	if math.Abs(lp-want) > 1e-10 {
		t.Errorf("LogProb(<s>, 東京) = %f, want %f", lp, want)
	}
}

func TestLogProb_Backoff(t *testing.T) {
	model, err := LoadARPA(strings.NewReader(testARPA))
	if err != nil {
		t.Fatalf("LoadARPA error: %v", err)
	}

	// P(東京 | タワー) -- no bigram exists, should backoff
	// backoff(タワー) + P_unigram(東京)
	lp := model.LogProb([]string{"タワー"}, "東京")
	backoff := -0.3 * math.Ln10
	unigramLP := -0.5 * math.Ln10
	want := backoff + unigramLP
	if math.Abs(lp-want) > 1e-10 {
		t.Errorf("LogProb(タワー, 東京) = %f, want %f", lp, want)
	}
}

func TestSentenceLogProb(t *testing.T) {
	model, err := LoadARPA(strings.NewReader(testARPA))
	if err != nil {
		t.Fatalf("LoadARPA error: %v", err)
	}

	lp := model.SentenceLogProb([]string{"東京", "タワー"})
	// P(<s>, 東京) + P(東京, タワー) + P(タワー, </s>)
	want := -0.3*math.Ln10 + -0.4*math.Ln10 + -0.2*math.Ln10
	if math.Abs(lp-want) > 1e-10 {
		t.Errorf("SentenceLogProb = %f, want %f", lp, want)
	}
}

func TestVocab(t *testing.T) {
	model, err := LoadARPA(strings.NewReader(testARPA))
	if err != nil {
		t.Fatalf("LoadARPA error: %v", err)
	}

	vocab := model.Vocab()
	if len(vocab) != 4 {
		t.Errorf("len(Vocab) = %d, want 4", len(vocab))
	}
}
