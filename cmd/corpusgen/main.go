package main

import (
	"fmt"
	"math/rand"
	"os"
	"strings"
)

var (
	persons = []string{
		"友達", "先生", "お母さん", "お父さん", "子供",
		"兄", "姉", "弟", "妹", "家族",
		"彼", "彼女", "みんな", "おじいさん", "おばあさん",
	}

	places = []string{
		"学校", "駅", "病院", "公園", "家",
		"部屋", "図書館", "レストラン", "カフェ", "会社",
		"教室", "店", "ホテル", "空港", "神社",
		"美術館", "体育館", "プール", "庭", "台所",
		"東京", "大阪", "京都", "北海道",
		"窓", "玄関", "屋上", "地下",
	}

	things = []string{
		"本", "新聞", "手紙", "写真", "映画",
		"音楽", "料理", "野菜", "魚", "パン",
		"花", "お茶", "ジュース", "水", "ごはん",
		"朝 ごはん", "昼 ごはん", "晩 ごはん", "弁当", "ケーキ",
		"荷物", "傘", "鍵", "財布", "靴",
		"服", "帽子", "眼鏡", "時計", "鞄",
		"紙", "ペン", "電話", "薬",
		"ピアノ", "ギター",
	}

	animals = []string{
		"犬", "猫", "鳥", "魚", "うさぎ",
		"馬", "虫",
	}

	nature = []string{
		"山", "海", "川", "空", "雨",
		"雪", "風", "花", "木", "森",
		"星", "月", "太陽",
	}

	times = []string{
		"朝", "昼", "夜", "夕方", "今日",
		"明日", "昨日", "毎日", "毎朝", "毎晩",
	}

	verbTransitive = []string{
		"見る", "食べる", "飲む", "読む", "書く",
		"作る", "買う", "撮る", "聴く", "洗う",
		"開ける", "閉める", "使う", "持つ", "送る",
		"運ぶ", "切る", "焼く", "弾く", "描く",
	}

	verbIntransitive = []string{
		"行く", "来る", "走る", "歩く", "泳ぐ",
		"飛ぶ", "寝る", "降る", "咲く", "鳴く",
		"遊ぶ", "笑う", "泣く", "座る", "立つ",
		"始まる", "終わる", "帰る", "起きる", "止まる",
	}

	verbMotion = []string{
		"行く", "来る", "帰る", "歩く", "走る",
	}

	adjectives = []string{
		"いい", "青い", "赤い", "大きい", "小さい",
		"おいしい", "暑い", "寒い", "高い", "安い",
		"新しい", "古い", "広い", "長い", "綺麗",
	}

	adverbs = []string{
		"とても", "もう", "まだ", "よく", "すぐ",
		"ゆっくり", "たくさん", "少し",
	}

	vehicles = []string{
		"電車", "バス", "自転車", "タクシー",
	}
)

type template struct {
	format string
	slots  [][]string
}

func main() {
	templates := []template{
		// THING を VERB_T
		{"%s を %s", [][]string{things, verbTransitive}},
		// PLACE で THING を VERB_T
		{"%s で %s を %s", [][]string{places, things, verbTransitive}},
		// PLACE に VERB_MOTION
		{"%s に %s", [][]string{places, verbMotion}},
		// PLACE へ VERB_MOTION
		{"%s へ %s", [][]string{places, verbMotion}},
		// PLACE まで VERB_MOTION
		{"%s まで %s", [][]string{places, verbMotion}},
		// PLACE から PLACE まで VERB_MOTION
		{"%s から %s まで %s", [][]string{places, places, verbMotion}},
		// PERSON と THING を VERB_T
		{"%s と %s を %s", [][]string{persons, things, verbTransitive}},
		// THING と THING を VERB_T
		{"%s と %s を %s", [][]string{things, things, verbTransitive}},
		// PERSON と VERB_I
		{"%s と %s", [][]string{persons, verbIntransitive}},
		// NATURE が VERB_I
		{"%s が %s", [][]string{nature, verbIntransitive}},
		// ANIMAL が PLACE で VERB_I
		{"%s が %s で %s", [][]string{animals, places, verbIntransitive}},
		// ANIMAL が VERB_I
		{"%s が %s", [][]string{animals, verbIntransitive}},
		// TIME から NATURE が VERB_I
		{"%s から %s が %s", [][]string{times, nature, verbIntransitive}},
		// TIME に THING を VERB_T
		{"%s に %s を %s", [][]string{times, things, verbTransitive}},
		// PLACE で PERSON と VERB_I
		{"%s で %s と %s", [][]string{places, persons, verbIntransitive}},
		// PLACE から VERB_MOTION
		{"%s から %s", [][]string{places, verbMotion}},
		// ADVERB THING を VERB_T
		{"%s %s を %s", [][]string{adverbs, things, verbTransitive}},
		// NATURE は ADJ
		{"%s は %s", [][]string{nature, adjectives}},
		// THING は ADJ
		{"%s は %s", [][]string{things, adjectives}},
		// PLACE は ADJ
		{"%s は %s", [][]string{places, adjectives}},
		// PERSON も VERB_I
		{"%s も %s", [][]string{persons, verbIntransitive}},
		// THING も VERB_T (使う形)
		{"%s も %s", [][]string{things, verbTransitive}},
		// PLACE で ANIMAL を VERB_T
		{"%s で %s を %s", [][]string{places, animals, verbTransitive}},
		// TIME PLACE に VERB_MOTION
		{"%s %s に %s", [][]string{times, places, verbMotion}},
		// TIME PLACE から VERB_MOTION
		{"%s %s から %s", [][]string{times, places, verbMotion}},
		// PLACE で NATURE を VERB_T
		{"%s で %s を %s", [][]string{places, nature, verbTransitive}},
		// NATURE まで VERB_MOTION (山まで歩く etc.)
		{"%s まで %s", [][]string{nature, verbMotion}},
		// NATURE から VERB_MOTION
		{"%s から %s", [][]string{nature, verbMotion}},
		// VEHICLE に 乗る / から 降りる patterns
		{"%s に 乗る", [][]string{vehicles}},
		{"%s から 降りる", [][]string{vehicles}},
		// PLACE から VEHICLE に 乗る
		{"%s から %s に 乗る", [][]string{places, vehicles}},
		// PLACE で VEHICLE に 乗る
		{"%s で %s に 乗る", [][]string{places, vehicles}},
	}

	seen := make(map[string]bool)
	rng := rand.New(rand.NewSource(42))

	for _, tmpl := range templates {
		total := 1
		for _, s := range tmpl.slots {
			total *= len(s)
		}

		if total <= 500 {
			generateAll(tmpl, seen)
		} else {
			target := 300
			if total > 5000 {
				target = 200
			}
			generateSampled(tmpl, seen, rng, target)
		}
	}

	// Print stats to stderr
	words := make(map[string]bool)
	for sent := range seen {
		for _, w := range strings.Fields(sent) {
			words[w] = true
		}
	}
	fmt.Fprintf(os.Stderr, "Generated %d sentences, %d unique words\n", len(seen), len(words))
}

func generateAll(tmpl template, seen map[string]bool) {
	nSlots := len(tmpl.slots)
	indices := make([]int, nSlots)

	for {
		args := make([]interface{}, nSlots)
		for i, idx := range indices {
			args[i] = tmpl.slots[i][idx]
		}
		sent := fmt.Sprintf(tmpl.format, args...)
		emit(sent, seen)

		carry := true
		for i := nSlots - 1; i >= 0 && carry; i-- {
			indices[i]++
			if indices[i] < len(tmpl.slots[i]) {
				carry = false
			} else {
				indices[i] = 0
			}
		}
		if carry {
			break
		}
	}
}

func generateSampled(tmpl template, seen map[string]bool, rng *rand.Rand, target int) {
	nSlots := len(tmpl.slots)
	attempts := target * 3

	for a := 0; a < attempts && target > 0; a++ {
		args := make([]interface{}, nSlots)
		for i := range tmpl.slots {
			args[i] = tmpl.slots[i][rng.Intn(len(tmpl.slots[i]))]
		}
		sent := fmt.Sprintf(tmpl.format, args...)
		if !seen[sent] {
			if emit(sent, seen) {
				target--
			}
		}
	}
}

func emit(sent string, seen map[string]bool) bool {
	words := strings.Fields(sent)
	normalized := strings.Join(words, " ")
	if seen[normalized] {
		return false
	}
	// Skip self-referencing (e.g., "魚 と 魚 を 買う")
	if len(words) >= 4 && words[0] == words[2] {
		return false
	}
	seen[normalized] = true
	fmt.Println(normalized)
	return true
}
