package acoustic

import (
	"testing"
)

func TestMakeTriphone(t *testing.T) {
	tri := MakeTriphone("i", "k", "u")
	if tri != "i-k+u" {
		t.Errorf("MakeTriphone(i,k,u) = %q, want %q", tri, "i-k+u")
	}
	tri = MakeTriphone("#", "a", "#")
	if tri != "#-a+#" {
		t.Errorf("MakeTriphone(#,a,#) = %q, want %q", tri, "#-a+#")
	}
}

func TestCenterPhoneme(t *testing.T) {
	tests := []struct {
		tri  Triphone
		want Phoneme
	}{
		{"i-k+u", PhonK},
		{"#-a+#", PhonA},
		{"a-sh+i", PhonSh},
		{"k-long+#", PhonLong},
		{"#-ng+g", PhonNg},
	}
	for _, tt := range tests {
		got := tt.tri.CenterPhoneme()
		if got != tt.want {
			t.Errorf("CenterPhoneme(%q) = %q, want %q", tt.tri, got, tt.want)
		}
	}
}

func TestCenterPhoneme_Fallback(t *testing.T) {
	// If no dash/plus, treat as monophone
	tri := Triphone("k")
	got := tri.CenterPhoneme()
	if got != PhonK {
		t.Errorf("CenterPhoneme(%q) = %q, want %q", tri, got, PhonK)
	}
}

func TestWordToTriphones(t *testing.T) {
	tests := []struct {
		name     string
		phonemes []Phoneme
		want     []Triphone
	}{
		{
			name:     "行く i-k-u",
			phonemes: []Phoneme{PhonI, PhonK, PhonU},
			want:     []Triphone{"#-i+k", "i-k+u", "k-u+#"},
		},
		{
			name:     "single phoneme",
			phonemes: []Phoneme{PhonA},
			want:     []Triphone{"#-a+#"},
		},
		{
			name:     "two phonemes",
			phonemes: []Phoneme{PhonK, PhonA},
			want:     []Triphone{"#-k+a", "k-a+#"},
		},
		{
			name:     "empty",
			phonemes: nil,
			want:     nil,
		},
		{
			name:     "multi-char phonemes sh-i-ng",
			phonemes: []Phoneme{PhonSh, PhonI, PhonNg},
			want:     []Triphone{"#-sh+i", "sh-i+ng", "i-ng+#"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := WordToTriphones(tt.phonemes)
			if len(got) != len(tt.want) {
				t.Fatalf("len = %d, want %d", len(got), len(tt.want))
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Errorf("[%d] = %q, want %q", i, got[i], tt.want[i])
				}
			}
		})
	}
}
