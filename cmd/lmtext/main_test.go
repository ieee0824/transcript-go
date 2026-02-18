package main

import (
	"os/exec"
	"testing"
)

func TestAllInDict(t *testing.T) {
	wordSet := map[string]bool{"東京": true, "に": true, "行く": true, "タワー": true}

	tests := []struct {
		words []string
		want  bool
	}{
		{[]string{"東京", "に", "行く"}, true},
		{[]string{"東京", "タワー", "に", "行く"}, true},
		{[]string{"東京", "から", "行く"}, false}, // "から" not in dict
		{[]string{"大阪"}, false},                  // OOV word
		{[]string{}, true},                         // empty is ok
	}

	for _, tt := range tests {
		got := allInDict(tt.words, wordSet)
		if got != tt.want {
			t.Errorf("allInDict(%v) = %v, want %v", tt.words, got, tt.want)
		}
	}
}

func TestSplitSentences(t *testing.T) {
	tests := []struct {
		input string
		want  int
	}{
		{"東京に行く。大阪に行く。", 2},
		{"東京に行く", 1},
		{"", 0},
		{"。。。", 0},
		{"東京に行く。", 1},
	}

	for _, tt := range tests {
		got := splitSentences(tt.input)
		if len(got) != tt.want {
			t.Errorf("splitSentences(%q) = %v (len=%d), want len=%d", tt.input, got, len(got), tt.want)
		}
	}
}

func TestMecabBatch(t *testing.T) {
	if _, err := exec.LookPath("mecab"); err != nil {
		t.Skip("MeCab not installed")
	}

	lines := []string{"東京タワーに行く", "お茶を飲む"}
	result, err := mecabBatch(lines)
	if err != nil {
		t.Fatal(err)
	}

	if len(result) != 2 {
		t.Fatalf("expected 2 results, got %d", len(result))
	}

	// Each line should produce at least 2 tokens
	for i, words := range result {
		if len(words) < 2 {
			t.Errorf("line %d: expected >= 2 words, got %v", i, words)
		}
	}
}
