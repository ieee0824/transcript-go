package lexicon

import "github.com/ieee0824/transcript-go/acoustic"

// p is a shorthand to build a phoneme slice.
func p(ps ...acoustic.Phoneme) []acoustic.Phoneme { return ps }

// kanaPhonemes maps katakana strings to phoneme sequences.
// Two-character entries (yōon) are checked before single characters (longest match).
var kanaPhonemes = []struct {
	kana     string
	phonemes []acoustic.Phoneme
}{
	// 拗音 (2文字) — must come before single-char entries
	{"キャ", p(acoustic.PhonK, acoustic.PhonY, acoustic.PhonA)},
	{"キュ", p(acoustic.PhonK, acoustic.PhonY, acoustic.PhonU)},
	{"キョ", p(acoustic.PhonK, acoustic.PhonY, acoustic.PhonO)},
	{"ギャ", p(acoustic.PhonG, acoustic.PhonY, acoustic.PhonA)},
	{"ギュ", p(acoustic.PhonG, acoustic.PhonY, acoustic.PhonU)},
	{"ギョ", p(acoustic.PhonG, acoustic.PhonY, acoustic.PhonO)},
	{"シャ", p(acoustic.PhonSh, acoustic.PhonA)},
	{"シュ", p(acoustic.PhonSh, acoustic.PhonU)},
	{"ショ", p(acoustic.PhonSh, acoustic.PhonO)},
	{"ジャ", p(acoustic.PhonJ, acoustic.PhonA)},
	{"ジュ", p(acoustic.PhonJ, acoustic.PhonU)},
	{"ジョ", p(acoustic.PhonJ, acoustic.PhonO)},
	{"チャ", p(acoustic.PhonCh, acoustic.PhonA)},
	{"チュ", p(acoustic.PhonCh, acoustic.PhonU)},
	{"チョ", p(acoustic.PhonCh, acoustic.PhonO)},
	{"ニャ", p(acoustic.PhonN, acoustic.PhonY, acoustic.PhonA)},
	{"ニュ", p(acoustic.PhonN, acoustic.PhonY, acoustic.PhonU)},
	{"ニョ", p(acoustic.PhonN, acoustic.PhonY, acoustic.PhonO)},
	{"ヒャ", p(acoustic.PhonH, acoustic.PhonY, acoustic.PhonA)},
	{"ヒュ", p(acoustic.PhonH, acoustic.PhonY, acoustic.PhonU)},
	{"ヒョ", p(acoustic.PhonH, acoustic.PhonY, acoustic.PhonO)},
	{"ビャ", p(acoustic.PhonB, acoustic.PhonY, acoustic.PhonA)},
	{"ビュ", p(acoustic.PhonB, acoustic.PhonY, acoustic.PhonU)},
	{"ビョ", p(acoustic.PhonB, acoustic.PhonY, acoustic.PhonO)},
	{"ピャ", p(acoustic.PhonP, acoustic.PhonY, acoustic.PhonA)},
	{"ピュ", p(acoustic.PhonP, acoustic.PhonY, acoustic.PhonU)},
	{"ピョ", p(acoustic.PhonP, acoustic.PhonY, acoustic.PhonO)},
	{"ミャ", p(acoustic.PhonM, acoustic.PhonY, acoustic.PhonA)},
	{"ミュ", p(acoustic.PhonM, acoustic.PhonY, acoustic.PhonU)},
	{"ミョ", p(acoustic.PhonM, acoustic.PhonY, acoustic.PhonO)},
	{"リャ", p(acoustic.PhonR, acoustic.PhonY, acoustic.PhonA)},
	{"リュ", p(acoustic.PhonR, acoustic.PhonY, acoustic.PhonU)},
	{"リョ", p(acoustic.PhonR, acoustic.PhonY, acoustic.PhonO)},
	{"ティ", p(acoustic.PhonT, acoustic.PhonI)},
	{"ディ", p(acoustic.PhonD, acoustic.PhonI)},
	{"ファ", p(acoustic.PhonF, acoustic.PhonA)},
	{"フィ", p(acoustic.PhonF, acoustic.PhonI)},
	{"フェ", p(acoustic.PhonF, acoustic.PhonE)},
	{"フォ", p(acoustic.PhonF, acoustic.PhonO)},
	{"フュ", p(acoustic.PhonF, acoustic.PhonY, acoustic.PhonU)},
	// 外来語拗音
	{"チェ", p(acoustic.PhonCh, acoustic.PhonE)},
	{"シェ", p(acoustic.PhonSh, acoustic.PhonE)},
	{"ジェ", p(acoustic.PhonJ, acoustic.PhonE)},
	{"ウィ", p(acoustic.PhonU, acoustic.PhonI)},
	{"ウェ", p(acoustic.PhonU, acoustic.PhonE)},
	{"ウォ", p(acoustic.PhonU, acoustic.PhonO)},
	{"ヴァ", p(acoustic.PhonB, acoustic.PhonA)},
	{"ヴィ", p(acoustic.PhonB, acoustic.PhonI)},
	{"ヴェ", p(acoustic.PhonB, acoustic.PhonE)},
	{"ヴォ", p(acoustic.PhonB, acoustic.PhonO)},
	{"トゥ", p(acoustic.PhonT, acoustic.PhonU)},
	{"ドゥ", p(acoustic.PhonD, acoustic.PhonU)},
	{"デュ", p(acoustic.PhonD, acoustic.PhonY, acoustic.PhonU)},
	{"テュ", p(acoustic.PhonT, acoustic.PhonY, acoustic.PhonU)},
	{"ツァ", p(acoustic.PhonTs, acoustic.PhonA)},
	{"ツィ", p(acoustic.PhonTs, acoustic.PhonI)},
	{"ツェ", p(acoustic.PhonTs, acoustic.PhonE)},
	{"ツォ", p(acoustic.PhonTs, acoustic.PhonO)},
	{"イェ", p(acoustic.PhonI, acoustic.PhonE)},
	{"クァ", p(acoustic.PhonK, acoustic.PhonW, acoustic.PhonA)},
	{"グァ", p(acoustic.PhonG, acoustic.PhonW, acoustic.PhonA)},

	// 単独カナ
	// ア行
	{"ア", p(acoustic.PhonA)},
	{"イ", p(acoustic.PhonI)},
	{"ウ", p(acoustic.PhonU)},
	{"エ", p(acoustic.PhonE)},
	{"オ", p(acoustic.PhonO)},
	// カ行
	{"カ", p(acoustic.PhonK, acoustic.PhonA)},
	{"キ", p(acoustic.PhonK, acoustic.PhonI)},
	{"ク", p(acoustic.PhonK, acoustic.PhonU)},
	{"ケ", p(acoustic.PhonK, acoustic.PhonE)},
	{"コ", p(acoustic.PhonK, acoustic.PhonO)},
	// ガ行
	{"ガ", p(acoustic.PhonG, acoustic.PhonA)},
	{"ギ", p(acoustic.PhonG, acoustic.PhonI)},
	{"グ", p(acoustic.PhonG, acoustic.PhonU)},
	{"ゲ", p(acoustic.PhonG, acoustic.PhonE)},
	{"ゴ", p(acoustic.PhonG, acoustic.PhonO)},
	// サ行
	{"サ", p(acoustic.PhonS, acoustic.PhonA)},
	{"シ", p(acoustic.PhonSh, acoustic.PhonI)},
	{"ス", p(acoustic.PhonS, acoustic.PhonU)},
	{"セ", p(acoustic.PhonS, acoustic.PhonE)},
	{"ソ", p(acoustic.PhonS, acoustic.PhonO)},
	// ザ行
	{"ザ", p(acoustic.PhonZ, acoustic.PhonA)},
	{"ジ", p(acoustic.PhonJ, acoustic.PhonI)},
	{"ズ", p(acoustic.PhonZ, acoustic.PhonU)},
	{"ゼ", p(acoustic.PhonZ, acoustic.PhonE)},
	{"ゾ", p(acoustic.PhonZ, acoustic.PhonO)},
	// タ行
	{"タ", p(acoustic.PhonT, acoustic.PhonA)},
	{"チ", p(acoustic.PhonCh, acoustic.PhonI)},
	{"ツ", p(acoustic.PhonTs, acoustic.PhonU)},
	{"テ", p(acoustic.PhonT, acoustic.PhonE)},
	{"ト", p(acoustic.PhonT, acoustic.PhonO)},
	// ダ行
	{"ダ", p(acoustic.PhonD, acoustic.PhonA)},
	{"ヂ", p(acoustic.PhonJ, acoustic.PhonI)},
	{"ヅ", p(acoustic.PhonZ, acoustic.PhonU)},
	{"デ", p(acoustic.PhonD, acoustic.PhonE)},
	{"ド", p(acoustic.PhonD, acoustic.PhonO)},
	// ナ行
	{"ナ", p(acoustic.PhonN, acoustic.PhonA)},
	{"ニ", p(acoustic.PhonN, acoustic.PhonI)},
	{"ヌ", p(acoustic.PhonN, acoustic.PhonU)},
	{"ネ", p(acoustic.PhonN, acoustic.PhonE)},
	{"ノ", p(acoustic.PhonN, acoustic.PhonO)},
	// ハ行
	{"ハ", p(acoustic.PhonH, acoustic.PhonA)},
	{"ヒ", p(acoustic.PhonH, acoustic.PhonI)},
	{"フ", p(acoustic.PhonF, acoustic.PhonU)},
	{"ヘ", p(acoustic.PhonH, acoustic.PhonE)},
	{"ホ", p(acoustic.PhonH, acoustic.PhonO)},
	// バ行
	{"バ", p(acoustic.PhonB, acoustic.PhonA)},
	{"ビ", p(acoustic.PhonB, acoustic.PhonI)},
	{"ブ", p(acoustic.PhonB, acoustic.PhonU)},
	{"ベ", p(acoustic.PhonB, acoustic.PhonE)},
	{"ボ", p(acoustic.PhonB, acoustic.PhonO)},
	// パ行
	{"パ", p(acoustic.PhonP, acoustic.PhonA)},
	{"ピ", p(acoustic.PhonP, acoustic.PhonI)},
	{"プ", p(acoustic.PhonP, acoustic.PhonU)},
	{"ペ", p(acoustic.PhonP, acoustic.PhonE)},
	{"ポ", p(acoustic.PhonP, acoustic.PhonO)},
	// マ行
	{"マ", p(acoustic.PhonM, acoustic.PhonA)},
	{"ミ", p(acoustic.PhonM, acoustic.PhonI)},
	{"ム", p(acoustic.PhonM, acoustic.PhonU)},
	{"メ", p(acoustic.PhonM, acoustic.PhonE)},
	{"モ", p(acoustic.PhonM, acoustic.PhonO)},
	// ヤ行
	{"ヤ", p(acoustic.PhonY, acoustic.PhonA)},
	{"ユ", p(acoustic.PhonY, acoustic.PhonU)},
	{"ヨ", p(acoustic.PhonY, acoustic.PhonO)},
	// ラ行
	{"ラ", p(acoustic.PhonR, acoustic.PhonA)},
	{"リ", p(acoustic.PhonR, acoustic.PhonI)},
	{"ル", p(acoustic.PhonR, acoustic.PhonU)},
	{"レ", p(acoustic.PhonR, acoustic.PhonE)},
	{"ロ", p(acoustic.PhonR, acoustic.PhonO)},
	// ワ行
	{"ワ", p(acoustic.PhonW, acoustic.PhonA)},
	{"ヲ", p(acoustic.PhonO)},
	// 小文字母音 (外来語フォールバック)
	{"ァ", p(acoustic.PhonA)},
	{"ィ", p(acoustic.PhonI)},
	{"ゥ", p(acoustic.PhonU)},
	{"ェ", p(acoustic.PhonE)},
	{"ォ", p(acoustic.PhonO)},
	// 特殊
	{"ン", p(acoustic.PhonNg)},
	{"ッ", p(acoustic.PhonQ)},
	{"ー", p(acoustic.PhonLong)},
	// ヴ (外来語)
	{"ヴ", p(acoustic.PhonB, acoustic.PhonU)},
}

// kanaMap indexes single and multi-rune kana for fast lookup.
// Built at init time from kanaPhonemes.
var kanaMap2 map[string][]acoustic.Phoneme // 2-char entries
var kanaMap1 map[string][]acoustic.Phoneme // 1-char entries

func init() {
	kanaMap2 = make(map[string][]acoustic.Phoneme)
	kanaMap1 = make(map[string][]acoustic.Phoneme)
	for _, e := range kanaPhonemes {
		runes := []rune(e.kana)
		if len(runes) == 2 {
			kanaMap2[e.kana] = e.phonemes
		} else {
			kanaMap1[e.kana] = e.phonemes
		}
	}
}

// KanaToPhonemes converts a katakana string to a phoneme sequence.
// Unknown characters are silently skipped.
func KanaToPhonemes(kana string) []acoustic.Phoneme {
	runes := []rune(kana)
	var result []acoustic.Phoneme
	for i := 0; i < len(runes); {
		// Try 2-char match first (longest match)
		if i+1 < len(runes) {
			key := string(runes[i : i+2])
			if ph, ok := kanaMap2[key]; ok {
				result = append(result, ph...)
				i += 2
				continue
			}
		}
		// Single-char match
		key := string(runes[i : i+1])
		if ph, ok := kanaMap1[key]; ok {
			result = append(result, ph...)
		}
		i++
	}
	return result
}
