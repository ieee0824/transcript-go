package language

import (
	"bytes"
	"fmt"
	"math/rand"
	"testing"
)

// buildTestCorpus generates a synthetic corpus with the given vocabulary size and sentence count.
func buildTestCorpus(vocabSize, sentenceCount, maxLen int) [][]string {
	rng := rand.New(rand.NewSource(42))
	vocab := make([]string, vocabSize)
	for i := range vocab {
		vocab[i] = fmt.Sprintf("w%d", i)
	}
	sentences := make([][]string, sentenceCount)
	for i := range sentences {
		n := 2 + rng.Intn(maxLen-1) // 2..maxLen words
		s := make([]string, n)
		for j := range s {
			s[j] = vocab[rng.Intn(vocabSize)]
		}
		sentences[i] = s
	}
	return sentences
}

func BenchmarkWriteARPA_Bigram_Small(b *testing.B) {
	// ~500 vocab, 1000 sentences — small baseline
	corpus := buildTestCorpus(500, 1000, 8)
	builder := NewBuilder(2)
	for _, s := range corpus {
		builder.AddSentence(s)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var buf bytes.Buffer
		builder.WriteARPA(&buf)
	}
}

func BenchmarkWriteARPA_Trigram_Small(b *testing.B) {
	// ~500 vocab, 1000 sentences
	corpus := buildTestCorpus(500, 1000, 8)
	builder := NewBuilder(3)
	for _, s := range corpus {
		builder.AddSentence(s)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var buf bytes.Buffer
		builder.WriteARPA(&buf)
	}
}

func BenchmarkWriteARPA_Trigram_Medium(b *testing.B) {
	// ~2000 vocab, 5000 sentences — closer to real use
	corpus := buildTestCorpus(2000, 5000, 10)
	builder := NewBuilder(3)
	for _, s := range corpus {
		builder.AddSentence(s)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var buf bytes.Buffer
		builder.WriteARPA(&buf)
	}
}

func BenchmarkWriteARPA_Trigram_Large(b *testing.B) {
	// ~5000 vocab, 20000 sentences — stress test
	corpus := buildTestCorpus(5000, 20000, 12)
	builder := NewBuilder(3)
	for _, s := range corpus {
		builder.AddSentence(s)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var buf bytes.Buffer
		builder.WriteARPA(&buf)
	}
}
