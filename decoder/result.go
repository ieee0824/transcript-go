package decoder

// Result holds the recognition output.
type Result struct {
	Text     string   // recognized text
	Words    []Word   // word-level details
	LogScore float64  // total log probability
	NBest    []Result // N-best alternatives (empty if NBestCount=0)
}

// Word holds per-word timing and score information.
type Word struct {
	Text       string
	StartFrame int
	EndFrame   int
	LogScore   float64
}
