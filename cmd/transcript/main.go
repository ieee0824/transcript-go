package main

import (
	"flag"
	"fmt"
	"os"
	transcript "github.com/ieee0824/transcript-go"
	"github.com/ieee0824/transcript-go/decoder"
)

func main() {
	amPath := flag.String("am", "", "path to acoustic model file")
	lmPath := flag.String("lm", "", "path to language model (ARPA format)")
	dictPath := flag.String("dict", "", "path to pronunciation dictionary")
	wavPath := flag.String("wav", "", "path to input WAV file")
	beam := flag.Float64("beam", 200.0, "beam width for decoding")
	lmWeight := flag.Float64("lm-weight", 10.0, "language model weight")
	wordPenalty := flag.Float64("word-penalty", 0.0, "word insertion penalty")
	maxTokens := flag.Int("max-tokens", 1000, "maximum active tokens")
	oovProb := flag.Float64("oov-prob", 0, "OOV unigram log10 probability (e.g. -5.0, 0=disable)")
	lmInterp := flag.Float64("lm-interp", 0.0, "LM interpolation weight with uniform prior (0=pure LM, 0.5=half uniform)")
	verbose := flag.Bool("v", false, "verbose output")

	flag.Parse()

	if *amPath == "" || *lmPath == "" || *dictPath == "" || *wavPath == "" {
		fmt.Fprintln(os.Stderr, "Usage: transcript -am MODEL -lm LM -dict DICT -wav AUDIO")
		flag.PrintDefaults()
		os.Exit(1)
	}

	rec, err := transcript.NewRecognizer(*amPath, *lmPath, *dictPath,
		transcript.WithDecoderConfig(decoder.Config{
			BeamWidth:            *beam,
			LMWeight:             *lmWeight,
			WordInsertionPenalty: *wordPenalty,
			MaxActiveTokens:      *maxTokens,
			LMInterpolation:     *lmInterp,
		}),
		transcript.WithOOVLogProb(*oovProb),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	result, err := rec.RecognizeFile(*wavPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	fmt.Println(result.Text)

	if *verbose {
		fmt.Fprintf(os.Stderr, "Score: %.4f\n", result.LogScore)
		for _, w := range result.Words {
			fmt.Fprintf(os.Stderr, "  [%d-%d] %s\n", w.StartFrame, w.EndFrame, w.Text)
		}
	}
}
