package lexicon

import (
	"strings"
	"testing"
	"github.com/ieee0824/transcript-go/acoustic"
)

const testDict = `# Japanese pronunciation dictionary
東京	トウキョウ	t o u k y o u
タワー	タワー	t a w a long
食べる	タベル	t a b e r u
食べる	タベル	t a b e r u
`

func TestLoadDict(t *testing.T) {
	d, err := Load(strings.NewReader(testDict))
	if err != nil {
		t.Fatalf("Load error: %v", err)
	}

	// 東京 should have 1 entry
	entries := d.Lookup("東京")
	if len(entries) != 1 {
		t.Fatalf("東京 entries = %d, want 1", len(entries))
	}
	if entries[0].Reading != "トウキョウ" {
		t.Errorf("東京 reading = %s, want トウキョウ", entries[0].Reading)
	}
	if len(entries[0].Phonemes) != 7 {
		t.Errorf("東京 phonemes = %d, want 7", len(entries[0].Phonemes))
	}
	if entries[0].Phonemes[0] != acoustic.PhonT {
		t.Errorf("東京 phonemes[0] = %s, want t", entries[0].Phonemes[0])
	}

	// 食べる should have 2 entries (duplicates)
	entries = d.Lookup("食べる")
	if len(entries) != 2 {
		t.Errorf("食べる entries = %d, want 2", len(entries))
	}
}

func TestPhonemeSequence(t *testing.T) {
	d, err := Load(strings.NewReader(testDict))
	if err != nil {
		t.Fatalf("Load error: %v", err)
	}

	phonemes, ok := d.PhonemeSequence("東京")
	if !ok {
		t.Fatal("東京 not found")
	}
	expected := []acoustic.Phoneme{"t", "o", "u", "k", "y", "o", "u"}
	if len(phonemes) != len(expected) {
		t.Fatalf("len = %d, want %d", len(phonemes), len(expected))
	}
	for i := range expected {
		if phonemes[i] != expected[i] {
			t.Errorf("phonemes[%d] = %s, want %s", i, phonemes[i], expected[i])
		}
	}
}

func TestLookupMissing(t *testing.T) {
	d, err := Load(strings.NewReader(testDict))
	if err != nil {
		t.Fatalf("Load error: %v", err)
	}

	_, ok := d.PhonemeSequence("存在しない")
	if ok {
		t.Error("should not find nonexistent word")
	}
}

func TestWords(t *testing.T) {
	d, err := Load(strings.NewReader(testDict))
	if err != nil {
		t.Fatalf("Load error: %v", err)
	}

	words := d.Words()
	if len(words) != 3 {
		t.Errorf("len(Words) = %d, want 3", len(words))
	}
}
