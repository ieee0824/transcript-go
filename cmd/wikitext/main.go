// wikitext extracts plain text sentences from a MediaWiki XML dump (bz2 compressed).
// Output: one sentence per line (split on 。), suitable for piping to lmtext.
package main

import (
	"bufio"
	"compress/bzip2"
	"encoding/xml"
	"flag"
	"fmt"
	"io"
	"os"
	"regexp"
	"strings"
)

// MediaWiki XML structure (minimal)
type page struct {
	Title    string   `xml:"title"`
	Revision revision `xml:"revision"`
}

type revision struct {
	Text string `xml:"text"`
}

var (
	// Remove MediaWiki markup patterns
	reRef       = regexp.MustCompile(`<ref[^>]*/>|<ref[^>]*>[\s\S]*?</ref>`)
	reHTML      = regexp.MustCompile(`<[^>]+>`)
	reBraces2   = regexp.MustCompile(`\{\{[^{}]*\}\}`)
	reBraces3   = regexp.MustCompile(`\{\{\{[^{}]*\}\}\}`)
	reBrackets  = regexp.MustCompile(`\[\[([^|\]]*\|)?([^\]]*)\]\]`)
	reExtLink   = regexp.MustCompile(`\[https?://[^\]]*\]`)
	reHeading   = regexp.MustCompile(`={2,}[^=]+=+`)
	reBullet    = regexp.MustCompile(`(?m)^[*#:;|!]+`)
	reMultiSpace = regexp.MustCompile(`\s+`)
	reCategory  = regexp.MustCompile(`\[\[(?:Category|カテゴリ):[^\]]+\]\]`)
	reFile      = regexp.MustCompile(`\[\[(?:File|ファイル|Image|画像):[^\]]+\]\]`)
)

func main() {
	maxPages := flag.Int("max-pages", 0, "maximum pages to process (0=all)")
	flag.Usage = func() {
		fmt.Fprintln(os.Stderr, "Usage: wikitext [options] jawiki-pages-articles.xml.bz2")
		fmt.Fprintln(os.Stderr, "  Extracts plain text from MediaWiki XML dump.")
		fmt.Fprintln(os.Stderr, "  Output: one sentence per line (split on 。)")
		fmt.Fprintln(os.Stderr)
		flag.PrintDefaults()
	}
	flag.Parse()

	if flag.NArg() == 0 {
		flag.Usage()
		os.Exit(1)
	}

	f, err := os.Open(flag.Arg(0))
	if err != nil {
		fmt.Fprintf(os.Stderr, "open: %v\n", err)
		os.Exit(1)
	}
	defer f.Close()

	var reader io.Reader
	if strings.HasSuffix(flag.Arg(0), ".bz2") {
		reader = bzip2.NewReader(f)
	} else {
		reader = f
	}

	writer := bufio.NewWriterSize(os.Stdout, 256*1024)
	defer writer.Flush()

	decoder := xml.NewDecoder(reader)
	pageCount := 0
	sentCount := 0

	for {
		tok, err := decoder.Token()
		if err == io.EOF {
			break
		}
		if err != nil {
			fmt.Fprintf(os.Stderr, "xml: %v\n", err)
			break
		}

		se, ok := tok.(xml.StartElement)
		if !ok || se.Name.Local != "page" {
			continue
		}

		var p page
		if err := decoder.DecodeElement(&p, &se); err != nil {
			continue
		}

		// Skip special pages
		if strings.Contains(p.Title, ":") {
			continue
		}

		text := cleanWikitext(p.Revision.Text)
		sentences := splitOnPeriod(text)
		for _, s := range sentences {
			s = strings.TrimSpace(s)
			if len(s) < 4 || len(s) > 200 {
				continue
			}
			fmt.Fprintln(writer, s)
			sentCount++
		}

		pageCount++
		if pageCount%10000 == 0 {
			fmt.Fprintf(os.Stderr, "\rPages: %d, Sentences: %d", pageCount, sentCount)
		}
		if *maxPages > 0 && pageCount >= *maxPages {
			break
		}
	}

	fmt.Fprintf(os.Stderr, "\rPages: %d, Sentences: %d\n", pageCount, sentCount)
}

func cleanWikitext(text string) string {
	// Remove templates (nested braces - do multiple passes)
	text = reBraces3.ReplaceAllString(text, "")
	for i := 0; i < 3; i++ {
		text = reBraces2.ReplaceAllString(text, "")
	}

	// Remove categories and file links
	text = reCategory.ReplaceAllString(text, "")
	text = reFile.ReplaceAllString(text, "")

	// Remove references
	text = reRef.ReplaceAllString(text, "")

	// Resolve wikilinks: [[target|display]] -> display, [[target]] -> target
	text = reBrackets.ReplaceAllString(text, "$2")

	// Remove external links
	text = reExtLink.ReplaceAllString(text, "")

	// Remove HTML tags
	text = reHTML.ReplaceAllString(text, "")

	// Remove headings
	text = reHeading.ReplaceAllString(text, "")

	// Remove bullet/list markers
	text = reBullet.ReplaceAllString(text, "")

	// Remove bold/italic
	text = strings.ReplaceAll(text, "'''", "")
	text = strings.ReplaceAll(text, "''", "")

	// Normalize whitespace
	text = reMultiSpace.ReplaceAllString(text, " ")

	return text
}

func splitOnPeriod(text string) []string {
	parts := strings.Split(text, "。")
	result := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			result = append(result, p)
		}
	}
	return result
}
