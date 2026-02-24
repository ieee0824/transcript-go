package lexicon

import "github.com/ieee0824/transcript-go/acoustic"

// katakanaToHiragana converts a katakana string to hiragana.
// Uses Unicode codepoint offset: katakana (U+30A0..U+30FF) → hiragana (U+3040..U+309F).
// Non-katakana characters are left unchanged.
func katakanaToHiragana(s string) string {
	runes := []rune(s)
	for i, r := range runes {
		if r >= 0x30A1 && r <= 0x30F6 { // ァ..ヶ
			runes[i] = r - 0x60
		} else if r == 0x30F3 { // ン
			runes[i] = 0x3093 // ん
		} else if r == 0x30C3 { // ッ
			runes[i] = 0x3063 // っ
		}
		// ー (0x30FC) has no hiragana equivalent; keep as-is
	}
	return string(runes)
}

// HiraganaEntries returns dictionary entries for hiragana fallback recognition.
// Each katakana entry in kanaPhonemes is converted to its hiragana equivalent.
// Includes single kana (あ,か,...), voiced (が,ざ,...), yōon (きゃ,しゅ,...),
// and special characters (ん,っ).
// The long vowel mark ー is excluded (no hiragana equivalent).
func HiraganaEntries() []Entry {
	var entries []Entry
	for _, kp := range kanaPhonemes {
		if kp.kana == "ー" {
			continue
		}
		hiragana := katakanaToHiragana(kp.kana)
		if hiragana == kp.kana {
			// No conversion happened (e.g. non-katakana); skip
			continue
		}
		phonemes := make([]acoustic.Phoneme, len(kp.phonemes))
		copy(phonemes, kp.phonemes)
		entries = append(entries, Entry{
			Word:     hiragana,
			Reading:  kp.kana,
			Phonemes: phonemes,
		})
	}
	return entries
}
