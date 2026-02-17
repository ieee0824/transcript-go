package acoustic

import (
	"encoding/gob"
	"io"
)

// AcousticModel holds all phoneme HMMs.
type AcousticModel struct {
	Phonemes   map[Phoneme]*PhonemeHMM
	Triphones  map[Triphone]*PhonemeHMM // context-dependent HMMs (nil for monophone-only models)
	FeatureDim int
	NumMix     int
}

// ResolveHMM returns the best HMM for a triphone: the triphone-specific HMM if it exists,
// otherwise falls back to the monophone HMM for the center phoneme.
func (am *AcousticModel) ResolveHMM(tri Triphone) *PhonemeHMM {
	if am.Triphones != nil {
		if hmm, ok := am.Triphones[tri]; ok {
			return hmm
		}
	}
	return am.Phonemes[tri.CenterPhoneme()]
}

// HasTriphones returns true if the model has any trained triphone HMMs.
func (am *AcousticModel) HasTriphones() bool {
	return len(am.Triphones) > 0
}

// ClonePhonemeHMM creates a deep copy of a PhonemeHMM.
func ClonePhonemeHMM(src *PhonemeHMM) *PhonemeHMM {
	dst := &PhonemeHMM{
		Phoneme:  src.Phoneme,
		States:   make([]*GMMState, NumStatesPerPhoneme),
		TransLog: make([][]float64, NumStatesPerPhoneme),
	}
	for i := range src.TransLog {
		dst.TransLog[i] = make([]float64, len(src.TransLog[i]))
		copy(dst.TransLog[i], src.TransLog[i])
	}
	for s := 1; s <= NumEmittingStates; s++ {
		if src.States[s] == nil {
			continue
		}
		srcGMM := src.States[s].GMM
		dstGMM := &GMM{
			Dim:        srcGMM.Dim,
			Components: make([]Gaussian, len(srcGMM.Components)),
		}
		for m, c := range srcGMM.Components {
			dstGMM.Components[m] = Gaussian{
				Mean:      append([]float64(nil), c.Mean...),
				Variance:  append([]float64(nil), c.Variance...),
				LogWeight: c.LogWeight,
			}
			dstGMM.Components[m].Precompute()
		}
		dstGMM.PrecomputeSoA()
		dst.States[s] = &GMMState{GMM: dstGMM}
	}
	return dst
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
	TriHMMs    map[string]serializedHMM // triphone HMMs (nil for monophone-only)
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
		sm.HMMs[string(pName)] = serializeHMM(hmm)
	}

	if len(am.Triphones) > 0 {
		sm.TriHMMs = make(map[string]serializedHMM, len(am.Triphones))
		for tri, hmm := range am.Triphones {
			sm.TriHMMs[string(tri)] = serializeHMM(hmm)
		}
	}

	return gob.NewEncoder(w).Encode(sm)
}

func serializeHMM(hmm *PhonemeHMM) serializedHMM {
	sh := serializedHMM{
		Phoneme:  string(hmm.Phoneme),
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
	return sh
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
		am.Phonemes[Phoneme(pStr)] = deserializeHMM(sh)
	}

	if len(sm.TriHMMs) > 0 {
		am.Triphones = make(map[Triphone]*PhonemeHMM, len(sm.TriHMMs))
		for triStr, sh := range sm.TriHMMs {
			am.Triphones[Triphone(triStr)] = deserializeHMM(sh)
		}
	}

	return am, nil
}

func deserializeHMM(sh serializedHMM) *PhonemeHMM {
	hmm := &PhonemeHMM{
		Phoneme:  Phoneme(sh.Phoneme),
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
	return hmm
}
