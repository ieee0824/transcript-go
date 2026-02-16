package language

import (
	"bufio"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"
)

// LoadARPA reads a language model in ARPA format.
// Log probabilities in ARPA files are base-10; they are converted to natural log.
func LoadARPA(r io.Reader) (*NGramModel, error) {
	scanner := bufio.NewScanner(r)
	model := NewNGramModel(1) // will be updated based on data

	// Skip until \data\ section
	for scanner.Scan() {
		if strings.TrimSpace(scanner.Text()) == "\\data\\" {
			break
		}
	}

	// Parse ngram counts
	maxOrder := 0
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		if strings.HasPrefix(line, "ngram ") {
			parts := strings.SplitN(line[6:], "=", 2)
			if len(parts) == 2 {
				order, _ := strconv.Atoi(strings.TrimSpace(parts[0]))
				if order > maxOrder {
					maxOrder = order
				}
			}
			continue
		}
		break
	}
	model.Order = maxOrder

	// Parse n-gram sections
	for {
		line := strings.TrimSpace(scanner.Text())

		if line == "\\end\\" {
			break
		}

		if strings.HasPrefix(line, "\\") && strings.HasSuffix(line, ":") {
			// e.g., \1-grams:
			orderStr := strings.TrimSuffix(strings.TrimPrefix(line, "\\"), "-grams:")
			order, err := strconv.Atoi(orderStr)
			if err != nil {
				// Skip this section header
				if !scanner.Scan() {
					break
				}
				continue
			}

			for scanner.Scan() {
				entry := strings.TrimSpace(scanner.Text())
				if entry == "" {
					continue
				}
				if strings.HasPrefix(entry, "\\") {
					break
				}
				if err := parseNGramLine(model, order, entry); err != nil {
					return nil, fmt.Errorf("parse n-gram line %q: %w", entry, err)
				}
			}
			continue
		}

		if !scanner.Scan() {
			break
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return model, nil
}

func parseNGramLine(model *NGramModel, order int, line string) error {
	fields := strings.Fields(line)
	if len(fields) < order+1 {
		return fmt.Errorf("too few fields for %d-gram: %q", order, line)
	}

	logProb, err := strconv.ParseFloat(fields[0], 64)
	if err != nil {
		return fmt.Errorf("parse log prob: %w", err)
	}
	// Convert base-10 to natural log
	logProb *= math.Ln10

	words := fields[1 : order+1]

	var logBackoff float64
	if len(fields) > order+1 {
		bo, err := strconv.ParseFloat(fields[order+1], 64)
		if err != nil {
			return fmt.Errorf("parse backoff: %w", err)
		}
		logBackoff = bo * math.Ln10
	}

	entry := ngramEntry{LogProb: logProb, LogBackoff: logBackoff}

	switch order {
	case 1:
		model.Unigrams[words[0]] = entry
	case 2:
		model.Bigrams[[2]string{words[0], words[1]}] = entry
	case 3:
		model.Trigrams[[3]string{words[0], words[1], words[2]}] = entry
	}

	return nil
}
