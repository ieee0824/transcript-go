package lexicon

import (
	"testing"

	"github.com/ieee0824/transcript-go/acoustic"
)

func TestKatakanaToHiragana(t *testing.T) {
	tests := []struct {
		in, want string
	}{
		{"ア", "あ"},
		{"カ", "か"},
		{"キャ", "きゃ"},
		{"ン", "ん"},
		{"ッ", "っ"},
		{"ー", "ー"}, // no hiragana equivalent
		{"シャ", "しゃ"},
		{"ガ", "が"},
		{"パ", "ぱ"},
	}
	for _, tt := range tests {
		got := katakanaToHiragana(tt.in)
		if got != tt.want {
			t.Errorf("katakanaToHiragana(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

func TestHiraganaEntries_Coverage(t *testing.T) {
	entries := HiraganaEntries()
	if len(entries) == 0 {
		t.Fatal("HiraganaEntries() returned empty slice")
	}

	// Should have at least basic kana (46) + voiced (20) + semi-voiced (5) + yōon
	if len(entries) < 70 {
		t.Errorf("HiraganaEntries() returned %d entries, expected at least 70", len(entries))
	}

	// Verify no duplicates
	seen := make(map[string]bool)
	for _, e := range entries {
		if seen[e.Word] {
			t.Errorf("duplicate hiragana entry: %q", e.Word)
		}
		seen[e.Word] = true
	}

	// Check specific entries
	check := map[string][]acoustic.Phoneme{
		"あ":  {acoustic.PhonA},
		"か":  {acoustic.PhonK, acoustic.PhonA},
		"し":  {acoustic.PhonSh, acoustic.PhonI},
		"ん":  {acoustic.PhonNg},
		"っ":  {acoustic.PhonQ},
		"きゃ": {acoustic.PhonK, acoustic.PhonY, acoustic.PhonA},
		"しゅ": {acoustic.PhonSh, acoustic.PhonU},
		"が":  {acoustic.PhonG, acoustic.PhonA},
		"ぱ":  {acoustic.PhonP, acoustic.PhonA},
	}
	entryMap := make(map[string][]acoustic.Phoneme)
	for _, e := range entries {
		entryMap[e.Word] = e.Phonemes
	}
	for word, wantPhon := range check {
		gotPhon, ok := entryMap[word]
		if !ok {
			t.Errorf("missing hiragana entry %q", word)
			continue
		}
		if len(gotPhon) != len(wantPhon) {
			t.Errorf("%q: got %d phonemes, want %d", word, len(gotPhon), len(wantPhon))
			continue
		}
		for i := range wantPhon {
			if gotPhon[i] != wantPhon[i] {
				t.Errorf("%q: phoneme[%d] = %q, want %q", word, i, gotPhon[i], wantPhon[i])
			}
		}
	}

	// ー should NOT be in the entries
	if _, ok := entryMap["ー"]; ok {
		t.Error("ー should not be in hiragana entries (no hiragana equivalent)")
	}
}

func TestHiraganaEntries_ReadingIsKatakana(t *testing.T) {
	for _, e := range HiraganaEntries() {
		if e.Reading == "" {
			t.Errorf("entry %q has empty reading", e.Word)
		}
		// Reading should be the original katakana
		if e.Reading == e.Word {
			t.Errorf("entry %q: reading should be katakana, got same as word", e.Word)
		}
	}
}

func TestAddHiraganaFallback(t *testing.T) {
	d := NewDictionary()
	d.Add("<sil>", "SIL", []acoustic.Phoneme{"sil"})
	d.Add("あ", "ア", []acoustic.Phoneme{acoustic.PhonA}) // already exists

	set := d.AddHiraganaFallback()

	// "あ" should NOT be in the returned set (already existed)
	if set["あ"] {
		t.Error("あ should not be in returned set (already in dict)")
	}

	// "か" should be in the set
	if !set["か"] {
		t.Error("か should be in returned set")
	}

	// "か" should now be in the dictionary
	entries := d.Lookup("か")
	if len(entries) == 0 {
		t.Error("か should be in dictionary after AddHiraganaFallback")
	}

	// "あ" should still have only 1 entry (not duplicated)
	if len(d.Lookup("あ")) != 1 {
		t.Errorf("あ should have 1 entry, got %d", len(d.Lookup("あ")))
	}
}
