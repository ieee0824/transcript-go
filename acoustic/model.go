package acoustic

import (
	"encoding/gob"
	"io"
)

// AcousticModel holds all phoneme HMMs.
type AcousticModel struct {
	Phonemes   map[Phoneme]*PhonemeHMM
	FeatureDim int
	NumMix     int
}

// NewAcousticModel creates an acoustic model with the Japanese phoneme set.
func NewAcousticModel(featureDim, numMix int) *AcousticModel {
	am := &AcousticModel{
		Phonemes:   make(map[Phoneme]*PhonemeHMM),
		FeatureDim: featureDim,
		NumMix:     numMix,
	}
	for _, p := range AllPhonemes() {
		am.Phonemes[p] = NewPhonemeHMM(p, featureDim, numMix)
	}
	return am
}

// serializable types for gob encoding
type serializedModel struct {
	FeatureDim int
	NumMix     int
	HMMs       map[string]serializedHMM
}

type serializedHMM struct {
	Phoneme  string
	TransLog [][]float64
	States   []serializedGMMState // only emitting states (indices 1..3)
}

type serializedGMMState struct {
	Components []serializedGaussian
	Dim        int
}

type serializedGaussian struct {
	Mean      []float64
	Variance  []float64
	LogWeight float64
}

// Save serializes the model to a writer using gob encoding.
func (am *AcousticModel) Save(w io.Writer) error {
	sm := serializedModel{
		FeatureDim: am.FeatureDim,
		NumMix:     am.NumMix,
		HMMs:       make(map[string]serializedHMM),
	}

	for pName, hmm := range am.Phonemes {
		sh := serializedHMM{
			Phoneme:  string(pName),
			TransLog: hmm.TransLog,
		}
		for i := 1; i <= NumEmittingStates; i++ {
			sg := serializedGMMState{
				Dim: hmm.States[i].GMM.Dim,
			}
			for _, c := range hmm.States[i].GMM.Components {
				sg.Components = append(sg.Components, serializedGaussian{
					Mean:      c.Mean,
					Variance:  c.Variance,
					LogWeight: c.LogWeight,
				})
			}
			sh.States = append(sh.States, sg)
		}
		sm.HMMs[string(pName)] = sh
	}

	return gob.NewEncoder(w).Encode(sm)
}

// Load deserializes an acoustic model from a reader.
func Load(r io.Reader) (*AcousticModel, error) {
	var sm serializedModel
	if err := gob.NewDecoder(r).Decode(&sm); err != nil {
		return nil, err
	}

	am := &AcousticModel{
		Phonemes:   make(map[Phoneme]*PhonemeHMM),
		FeatureDim: sm.FeatureDim,
		NumMix:     sm.NumMix,
	}

	for pStr, sh := range sm.HMMs {
		p := Phoneme(pStr)
		hmm := &PhonemeHMM{
			Phoneme:  p,
			States:   make([]*GMMState, NumStatesPerPhoneme),
			TransLog: sh.TransLog,
		}

		for i, sg := range sh.States {
			stateIdx := i + 1
			gmm := &GMM{Dim: sg.Dim}
			for _, sc := range sg.Components {
				g := Gaussian{
					Mean:      sc.Mean,
					Variance:  sc.Variance,
					LogWeight: sc.LogWeight,
				}
				g.Precompute()
				gmm.Components = append(gmm.Components, g)
			}
			gmm.PrecomputeSoA()
			hmm.States[stateIdx] = &GMMState{GMM: gmm}
		}

		am.Phonemes[p] = hmm
	}

	return am, nil
}
