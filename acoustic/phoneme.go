package acoustic

// Phoneme represents a Japanese phoneme.
type Phoneme string

const (
	// Silence and pause
	PhonSil  Phoneme = "sil"  // silence
	PhonSP   Phoneme = "sp"   // short pause

	// Vowels
	PhonA Phoneme = "a"
	PhonI Phoneme = "i"
	PhonU Phoneme = "u"
	PhonE Phoneme = "e"
	PhonO Phoneme = "o"

	// Stops (voiceless/voiced)
	PhonK Phoneme = "k"
	PhonG Phoneme = "g"
	PhonT Phoneme = "t"
	PhonD Phoneme = "d"
	PhonP Phoneme = "p"
	PhonB Phoneme = "b"

	// Fricatives
	PhonS  Phoneme = "s"
	PhonZ  Phoneme = "z"
	PhonH  Phoneme = "h"
	PhonF  Phoneme = "f"  // [ɸ] as in ふ

	// Affricates
	PhonCh Phoneme = "ch" // [tɕ] as in ち
	PhonTs Phoneme = "ts" // [ts] as in つ
	PhonJ  Phoneme = "j"  // [dʑ] as in じ

	// Nasals
	PhonM  Phoneme = "m"
	PhonN  Phoneme = "n"
	PhonNg Phoneme = "ng" // moraic nasal ん

	// Liquid
	PhonR Phoneme = "r" // Japanese flap

	// Glides
	PhonY Phoneme = "y"
	PhonW Phoneme = "w"

	// Sibilant
	PhonSh Phoneme = "sh" // [ɕ] as in し

	// Special morae
	PhonQ    Phoneme = "q"    // geminate っ
	PhonLong Phoneme = "long" // long vowel ー
)

// NumEmittingStates is the number of emitting states per phoneme HMM.
const NumEmittingStates = 3

// NumStatesPerPhoneme is the total states: entry + emitting + exit.
const NumStatesPerPhoneme = NumEmittingStates + 2

// AllPhonemes returns the complete Japanese phoneme set.
func AllPhonemes() []Phoneme {
	return []Phoneme{
		PhonSil, PhonSP,
		PhonA, PhonI, PhonU, PhonE, PhonO,
		PhonK, PhonG, PhonT, PhonD, PhonP, PhonB,
		PhonS, PhonZ, PhonH, PhonF,
		PhonCh, PhonTs, PhonJ,
		PhonM, PhonN, PhonNg,
		PhonR,
		PhonY, PhonW,
		PhonSh,
		PhonQ, PhonLong,
	}
}
