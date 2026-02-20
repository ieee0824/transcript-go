package lexicon

import (
	"testing"

	"github.com/ieee0824/transcript-go/acoustic"
)

func TestPhonemeEditDistance(t *testing.T) {
	p := func(ps ...acoustic.Phoneme) []acoustic.Phoneme { return ps }

	tests := []struct {
		name string
		a, b []acoustic.Phoneme
		want int
	}{
		{"identical", p(acoustic.PhonK, acoustic.PhonA), p(acoustic.PhonK, acoustic.PhonA), 0},
		{"empty_both", nil, nil, 0},
		{"empty_a", nil, p(acoustic.PhonA, acoustic.PhonI), 2},
		{"empty_b", p(acoustic.PhonA), nil, 1},
		{"substitution", p(acoustic.PhonK, acoustic.PhonA), p(acoustic.PhonG, acoustic.PhonA), 1},
		{"insertion", p(acoustic.PhonK, acoustic.PhonA), p(acoustic.PhonK, acoustic.PhonA, acoustic.PhonI), 1},
		{"deletion", p(acoustic.PhonK, acoustic.PhonA, acoustic.PhonI), p(acoustic.PhonK, acoustic.PhonA), 1},
		{
			"tori_vs_tori", // 取り vs 撮り (same phonemes)
			p(acoustic.PhonT, acoustic.PhonO, acoustic.PhonR, acoustic.PhonI),
			p(acoustic.PhonT, acoustic.PhonO, acoustic.PhonR, acoustic.PhonI),
			0,
		},
		{
			"kasa_vs_asa", // 傘 vs 朝
			p(acoustic.PhonK, acoustic.PhonA, acoustic.PhonS, acoustic.PhonA),
			p(acoustic.PhonA, acoustic.PhonS, acoustic.PhonA),
			1,
		},
		{
			"mike_vs_miku", // マイク vs ミク
			p(acoustic.PhonM, acoustic.PhonA, acoustic.PhonI, acoustic.PhonK, acoustic.PhonU),
			p(acoustic.PhonM, acoustic.PhonI, acoustic.PhonK, acoustic.PhonU),
			1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := PhonemeEditDistance(tt.a, tt.b)
			if got != tt.want {
				t.Errorf("PhonemeEditDistance() = %d, want %d", got, tt.want)
			}
		})
	}
}
