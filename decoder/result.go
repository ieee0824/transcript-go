package decoder

// Result holds the recognition output.
type Result struct {
	Text     string  // recognized text
	Words    []Word  // word-level details
	LogScore float64 // total log probability
}

// Word holds per-word timing and score information.
type Word struct {
	Text       string
	StartFrame int
	EndFrame   int
	LogScore   float64
}
