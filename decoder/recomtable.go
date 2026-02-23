package decoder

// recomTable is an open-addressing hash table for token recombination.
// Uses generation-based clearing for O(1) reset between frames.
type recomTable struct {
	keys []recomKey
	vals []int32
	gens []uint32
	gen  uint32
	mask int
	size int
}

func newRecomTable(hint int) *recomTable {
	// Capacity = next power of 2 >= hint*2 (load factor < 0.5)
	cap := 64
	for cap < hint*2 {
		cap <<= 1
	}
	return &recomTable{
		keys: make([]recomKey, cap),
		vals: make([]int32, cap),
		gens: make([]uint32, cap),
		gen:  1,
		mask: cap - 1,
	}
}

func (t *recomTable) clear() {
	t.gen++
	t.size = 0
	// On generation overflow (unlikely, every ~4 billion frames), reset all slots
	if t.gen == 0 {
		t.gen = 1
		for i := range t.gens {
			t.gens[i] = 0
		}
	}
}

func hashRecomKey(k recomKey) uint64 {
	h := uint64(k.nodeIdx)*0x9E3779B97F4A7C15 ^
		uint64(k.stateIdx)*0x517CC1B727220A95 ^
		uint64(k.lastWordID)*0x6C62272E07BB0142 ^
		uint64(k.prevWordID)*0x62B821756295C58D
	// Mix bits
	h ^= h >> 33
	h *= 0xFF51AFD7ED558CCD
	h ^= h >> 33
	return h
}

// lookup returns the value and true if the key exists, or 0 and false otherwise.
func (t *recomTable) lookup(key recomKey) (int32, bool) {
	h := hashRecomKey(key)
	pos := int(h) & t.mask
	for {
		if t.gens[pos] != t.gen {
			return 0, false
		}
		if t.keys[pos] == key {
			return t.vals[pos], true
		}
		pos = (pos + 1) & t.mask
	}
}

// insert adds a key-value pair. The key must not already exist.
func (t *recomTable) insert(key recomKey, val int32) {
	t.size++
	if t.size > (t.mask+1)/2 {
		t.grow()
	}
	h := hashRecomKey(key)
	pos := int(h) & t.mask
	for t.gens[pos] == t.gen {
		pos = (pos + 1) & t.mask
	}
	t.keys[pos] = key
	t.vals[pos] = val
	t.gens[pos] = t.gen
}

// update sets a new value for an existing key at the given position.
// lookupAndUpdate combines lookup + conditional update in one probe.
func (t *recomTable) lookupOrInsert(key recomKey, val int32) (int32, bool) {
	h := hashRecomKey(key)
	pos := int(h) & t.mask
	for {
		if t.gens[pos] != t.gen {
			// Empty slot — insert here
			t.size++
			if t.size > (t.mask+1)/2 {
				t.grow()
				return t.lookupOrInsert(key, val)
			}
			t.keys[pos] = key
			t.vals[pos] = val
			t.gens[pos] = t.gen
			return val, false
		}
		if t.keys[pos] == key {
			return t.vals[pos], true
		}
		pos = (pos + 1) & t.mask
	}
}

func (t *recomTable) grow() {
	oldKeys := t.keys
	oldVals := t.vals
	oldGens := t.gens
	oldGen := t.gen

	newCap := (t.mask + 1) * 2
	t.keys = make([]recomKey, newCap)
	t.vals = make([]int32, newCap)
	t.gens = make([]uint32, newCap)
	t.mask = newCap - 1
	t.gen++
	if t.gen == 0 {
		t.gen = 1
	}

	for i := range oldKeys {
		if oldGens[i] == oldGen {
			h := hashRecomKey(oldKeys[i])
			pos := int(h) & t.mask
			for t.gens[pos] == t.gen {
				pos = (pos + 1) & t.mask
			}
			t.keys[pos] = oldKeys[i]
			t.vals[pos] = oldVals[i]
			t.gens[pos] = t.gen
		}
	}
}
