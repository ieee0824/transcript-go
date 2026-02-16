package lexicon

import (
	"strings"
	"testing"

	"github.com/ieee0824/transcript-go/acoustic"
)

func phonemeStr(ps []acoustic.Phoneme) string {
	ss := make([]string, len(ps))
	for i, p := range ps {
		ss[i] = string(p)
	}
	return strings.Join(ss, " ")
}

func TestKanaToPhonemes(t *testing.T) {
	tests := []struct {
		kana string
		want string
	}{
		// 基本母音
		{"ア", "a"},
		{"アイウエオ", "a i u e o"},
		// カ行
		{"カキクケコ", "k a k i k u k e k o"},
		// 特殊子音
		{"シ", "sh i"},
		{"チ", "ch i"},
		{"ツ", "ts u"},
		{"フ", "f u"},
		// 拗音
		{"キャ", "k y a"},
		{"シャ", "sh a"},
		{"チャ", "ch a"},
		{"ニュ", "n y u"},
		{"リョ", "r y o"},
		// 促音・長音・撥音
		{"ッ", "q"},
		{"ー", "long"},
		{"ン", "ng"},
		// 複合テスト
		{"トウキョウ", "t o u k y o u"},
		{"タワー", "t a w a long"},
		{"タベル", "t a b e r u"},
		{"シンブン", "sh i ng b u ng"},
		{"ガッコウ", "g a q k o u"},
		{"オチャ", "o ch a"},
		{"テンキ", "t e ng k i"},
		// ヲ
		{"ヲ", "o"},
		// 外来語
		{"ファイル", "f a i r u"},
		{"ティー", "t i long"},
		{"ディスク", "d i s u k u"},
		// 空文字列
		{"", ""},
	}

	for _, tt := range tests {
		t.Run(tt.kana, func(t *testing.T) {
			got := phonemeStr(KanaToPhonemes(tt.kana))
			if got != tt.want {
				t.Errorf("KanaToPhonemes(%q) = %q, want %q", tt.kana, got, tt.want)
			}
		})
	}
}
