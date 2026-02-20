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
		"窓", "玄関", "屋上",
		"大学", "食堂", "工場",
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
		"目標", "食材", "忘れ物", "仕事",
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
		"午前", "午後",
	}

	// 「V-stem に 行く」パターン用の動詞語幹
	verbStems = []string{
		"買い", "取り", "食べ", "見", "遊び",
		"飲み", "泳ぎ", "撮り",
	}

	// 体の部分・抽象名詞（「が VERB_I」パターン用）
	bodyAbstract = []string{
		"お腹", "頭", "体", "気持ち", "準備",
	}

	verbTransitive = []string{
		"見る", "食べる", "飲む", "読む", "書く",
		"作る", "買う", "撮る", "聴く", "洗う",
		"開ける", "閉める", "使う", "持つ", "送る",
		"運ぶ", "切る", "焼く", "弾く", "描く",
		"着る", "決める",
	}

	verbIntransitive = []string{
		"行く", "来る", "走る", "歩く", "泳ぐ",
		"飛ぶ", "寝る", "降る", "咲く", "鳴く",
		"遊ぶ", "笑う", "泣く", "座る", "立つ",
		"始まる", "終わる", "帰る", "起きる", "止まる",
		"空く",
	}

	verbMotion = []string{
		"行く", "来る", "帰る", "歩く", "走る",
	}

	adjectives = []string{
		"いい", "青い", "赤い", "大きい", "小さい",
		"おいしい", "暑い", "寒い", "高い", "安い",
		"新しい", "古い", "広い", "長い", "きれい",
	}

	naAdj = []string{
		"きれい", "元気", "静か", "有名",
	}

	adverbs = []string{
		"とても", "もう", "まだ", "よく", "すぐ",
		"ゆっくり", "たくさん", "少し",
	}

	vehicles = []string{
		"電車", "バス", "自転車", "タクシー",
	}

	// === 大学・工学カテゴリ ===

	// 学科・科目
	subjects = []string{
		"数学", "物理", "化学", "英語", "情報",
	}

	// 大学で扱う対象物
	academicThings = []string{
		"レポート", "論文", "課題", "宿題", "卒論",
		"プログラム", "データ", "グラフ", "スライド",
		"教科書", "成績", "単位",
	}

	// 工学系の対象物
	engineeringThings = []string{
		"回路", "基板", "ロボット", "センサー", "モーター",
		"装置", "機械", "材料", "配線", "電池",
		"ケーブル", "電源", "信号",
	}

	// 研究室・実験室の機器（パソコン・スイッチは一般文との偽マッチ回避のため除外）
	equipment = []string{
		"プリンター", "モニター",
		"キーボード", "マウス", "黒板",
	}

	// 工学・学術活動が行われる場所（窓・玄関等の一般場所を含めない）
	labPlaces = []string{
		"大学", "教室", "図書館", "会社", "工場",
	}

	// 大学の人物
	academicPeople = []string{
		"教授", "先輩", "後輩", "学生", "院生",
	}

	// する動詞の名詞部分
	suruNouns = []string{
		"勉強", "研究", "実験", "計算", "測定",
		"分析", "設計", "観察", "記録", "報告",
		"発表", "提出", "製作", "比較", "調整",
	}

	// 工学系の他動詞（書く・読むはverbTransitiveと重複するため除外、動かすは偽マッチ回避）
	verbAcademic = []string{
		"解く", "調べる", "試す",
		"確かめる", "組み立てる", "組む", "繋ぐ",
		"直す", "測る",
	}

	// 工学系の自動詞（止まる・始まる・終わるはverbIntransitiveと重複、壊れるは偽マッチ回避）
	verbAcademicI = []string{
		"動く", "光る", "繋がる",
	}

	// === 外来語カテゴリ ===

	// 日用品・家電
	loanDaily = []string{
		"コーヒー", "ビール", "ジュース", "ミルク", "パン",
		"テレビ", "エアコン", "マイク", "カメラ", "ラジオ",
		"テーブル", "ソファ", "カーテン", "タオル", "コップ",
		"ナイフ", "フォーク", "スプーン", "ボトル", "バッグ",
		"シャツ", "ジャケット", "コート", "ブーツ", "サンダル",
	}

	// 場所・施設
	loanPlaces = []string{
		"コンビニ", "スーパー", "デパート", "ホテル", "レストラン",
		"カフェ", "ビル", "マンション", "アパート", "パーキング",
		"エレベーター", "エスカレーター", "トイレ", "ロビー",
	}

	// IT・テクノロジー
	loanTech = []string{
		"パソコン", "インターネット", "ソフト", "アプリ", "サーバー",
		"ネットワーク", "システム", "ファイル", "フォルダ", "メモリ",
		"ディスプレイ", "プリンター", "マウス", "キーボード",
	}

	// スポーツ・趣味
	loanHobby = []string{
		"サッカー", "テニス", "ゴルフ", "スキー", "ジョギング",
		"ゲーム", "ドラマ", "ニュース", "コンサート", "イベント",
	}

	// する動詞になる外来語
	loanSuru = []string{
		"チェック", "スタート", "ストップ", "キャンセル", "コピー",
		"プリント", "ダウンロード", "アップロード", "インストール",
		"クリック", "ログイン", "ログアウト", "サイン", "メモ",
		"リセット", "セット", "テスト", "チャレンジ", "トレーニング",
	}

	// === ビジネスカテゴリ ===

	// ビジネスで扱う対象物
	businessThings = []string{
		"資料", "書類", "名刺", "メール", "契約",
		"予算", "企画", "提案", "商品", "見積",
		"スケジュール", "プロジェクト", "マニュアル",
	}

	// ビジネスの人物（上司・部下は既存personsと重複しないよう注意）
	businessPeople = []string{
		"上司", "部下", "同僚", "社長", "課長",
		"部長", "担当",
	}

	// ビジネスする動詞の名詞部分
	businessSuru = []string{
		"確認", "準備", "報告", "連絡", "相談",
		"説明", "検討", "決定", "承認", "申請",
		"登録", "作成", "修正", "変更", "対応",
		"管理", "計画", "実行", "完了",
	}

	// 勤怠関連する動詞
	workSuru = []string{
		"出勤", "退勤", "出張", "残業", "休憩",
		"出席", "欠席",
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

		// === 日常生活テンプレート ===

		// PERSON が THING を VERB_T (妹 が ピアノ を 弾く)
		{"%s が %s を %s", [][]string{persons, things, verbTransitive}},
		// PERSON に THING を VERB_T (お母さん に 電話 を かける)
		{"%s に %s を %s", [][]string{persons, things, verbTransitive}},
		// PERSON と PLACE を VERB_MOTION (お父さん と 公園 を 歩く)
		{"%s と %s を %s", [][]string{persons, places, verbMotion}},
		// ADJ NATURE が VERB_I (赤い 花 が 咲く)
		{"%s %s が %s", [][]string{adjectives, nature, verbIntransitive}},
		// NATURE/THING が ADJ です (空 が きれい です)
		{"%s が %s です", [][]string{nature, naAdj}},
		{"%s が %s です", [][]string{things, naAdj}},
		// BODY/ABSTRACT が VERB_I (お腹 が 空く)
		{"%s が %s", [][]string{bodyAbstract, verbIntransitive}},
		// THING を VERB_STEM に VERB_MOTION (忘れ物 を 取り に 行く)
		{"%s を %s に %s", [][]string{things, verbStems, verbMotion}},
		// TIME から THING を VERB_T (午後 から 仕事 を する)
		{"%s から %s を %s", [][]string{times, things, verbTransitive}},
		// PERSON が ADVERB に VERB_I (子供 が 元気 に 遊ぶ)
		{"%s が 元気 に %s", [][]string{persons, verbIntransitive}},
		// PERSON が PLACE で SURU する (学生 が 教室 で 勉強 する)
		{"%s が %s で %s する", [][]string{academicPeople, labPlaces, suruNouns}},
		// PERSON に SURU する (先生 に 説明 する) -- also persons
		{"%s に %s する", [][]string{persons, businessSuru}},
		// BUSINESS_SURU が BUSINESS_SURU する (準備 が 完了 する)
		{"%s が %s する", [][]string{businessSuru, businessSuru}},

		// === 大学・工学テンプレート ===

		// SUBJECT を 勉強 する
		{"%s を 勉強 する", [][]string{subjects}},
		// SUBJECT を 研究 する
		{"%s を 研究 する", [][]string{subjects}},
		// ACADEMIC_THING を VERB_ACADEMIC (レポート を 書く, 課題 を 解く)
		{"%s を %s", [][]string{academicThings, verbAcademic}},
		// ACADEMIC_THING を SURU する (データ を 分析 する)
		{"%s を %s する", [][]string{academicThings, suruNouns}},
		// ENGINEERING_THING を SURU する (回路 を 設計 する)
		{"%s を %s する", [][]string{engineeringThings, suruNouns}},
		// ENGINEERING_THING を VERB_ACADEMIC (回路 を 組む)
		{"%s を %s", [][]string{engineeringThings, verbAcademic}},
		// ENGINEERING_THING が VERB_ACADEMIC_I (機械 が 動く)
		{"%s が %s", [][]string{engineeringThings, verbAcademicI}},
		// EQUIPMENT を 使う
		{"%s を 使う", [][]string{equipment}},
		// EQUIPMENT を VERB_ACADEMIC
		{"%s を %s", [][]string{equipment, verbAcademic}},
		// EQUIPMENT が VERB_ACADEMIC_I
		{"%s が %s", [][]string{equipment, verbAcademicI}},
		// ACADEMIC_PERSON と SURU する (教授 と 研究 する)
		{"%s と %s する", [][]string{academicPeople, suruNouns}},
		// ACADEMIC_PERSON と ACADEMIC_THING を VERB_ACADEMIC
		{"%s と %s を %s", [][]string{academicPeople, academicThings, verbAcademic}},
		// ACADEMIC_PERSON が SURU する (教授 が 発表 する)
		{"%s が %s する", [][]string{academicPeople, suruNouns}},
		// SUBJECT の ACADEMIC_THING を VERB_ACADEMIC (数学 の 課題 を 解く)
		{"%s の %s を %s", [][]string{subjects, academicThings, verbAcademic}},
		// SUBJECT の ACADEMIC_THING を SURU する (物理 の 実験 を 記録 する)
		{"%s の %s を %s する", [][]string{subjects, academicThings, suruNouns}},
		// TIME に ACADEMIC_THING を SURU する
		{"%s に %s を %s する", [][]string{times, academicThings, suruNouns}},
		// TIME に ACADEMIC_THING を VERB_ACADEMIC
		{"%s に %s を %s", [][]string{times, academicThings, verbAcademic}},
		// ACADEMIC_PERSON に SURU する (教授 に 報告 する)
		{"%s に %s する", [][]string{academicPeople, suruNouns}},
		// ACADEMIC_PERSON も SURU する
		{"%s も %s する", [][]string{academicPeople, suruNouns}},

		// === ビジネステンプレート ===

		// BUSINESS_THING を BUSINESS_SURU する (資料 を 作成 する)
		{"%s を %s する", [][]string{businessThings, businessSuru}},
		// BUSINESS_THING を VERB_T (書類 を 読む, メール を 送る)
		{"%s を %s", [][]string{businessThings, verbTransitive}},
		// BUSINESS_PERSON と BUSINESS_SURU する (上司 と 相談 する)
		{"%s と %s する", [][]string{businessPeople, businessSuru}},
		// BUSINESS_PERSON に BUSINESS_SURU する (部長 に 報告 する)
		{"%s に %s する", [][]string{businessPeople, businessSuru}},
		// BUSINESS_PERSON が BUSINESS_SURU する (課長 が 承認 する)
		{"%s が %s する", [][]string{businessPeople, businessSuru}},
		// BUSINESS_PERSON と BUSINESS_THING を BUSINESS_SURU する
		{"%s と %s を %s する", [][]string{businessPeople, businessThings, businessSuru}},
		// TIME に BUSINESS_SURU する (明日 に 出張 する)
		{"%s に %s する", [][]string{times, workSuru}},
		// TIME BUSINESS_SURU する (今日 残業 する)
		{"%s %s する", [][]string{times, workSuru}},
		// BUSINESS_THING の BUSINESS_SURU する (予算 の 確認 する... → 予算 を 確認 する is better)
		// BUSINESS_PERSON も BUSINESS_SURU する
		{"%s も %s する", [][]string{businessPeople, businessSuru}},
		// ADVERB BUSINESS_THING を BUSINESS_SURU する
		{"%s %s を %s する", [][]string{adverbs, businessThings, businessSuru}},

		// === 外来語テンプレート ===

		// LOAN_DAILY を VERB_T (コーヒー を 飲む, カメラ を 買う)
		{"%s を %s", [][]string{loanDaily, verbTransitive}},
		// LOAN_DAILY を LOAN_SURU する (マイク を チェック する)
		{"%s を %s する", [][]string{loanDaily, loanSuru}},
		// LOAN_PLACES で THING を VERB_T (コンビニ で パン を 買う)
		{"%s で %s を %s", [][]string{loanPlaces, things, verbTransitive}},
		// LOAN_PLACES に VERB_MOTION (スーパー に 行く)
		{"%s に %s", [][]string{loanPlaces, verbMotion}},
		// LOAN_PLACES へ VERB_MOTION
		{"%s へ %s", [][]string{loanPlaces, verbMotion}},
		// LOAN_PLACES で LOAN_DAILY を VERB_T (カフェ で コーヒー を 飲む)
		{"%s で %s を %s", [][]string{loanPlaces, loanDaily, verbTransitive}},
		// LOAN_TECH を VERB_T (パソコン を 使う)
		{"%s を %s", [][]string{loanTech, verbTransitive}},
		// LOAN_TECH を LOAN_SURU する (ソフト を インストール する)
		{"%s を %s する", [][]string{loanTech, loanSuru}},
		// LOAN_SURU する (チェック する, スタート する)
		{"%s する", [][]string{loanSuru}},
		// LOAN_HOBBY を VERB_T (サッカー を 見る)
		{"%s を %s", [][]string{loanHobby, verbTransitive}},
		// PLACE で LOAN_HOBBY を VERB_T (公園 で サッカー を する)
		{"%s で %s を %s", [][]string{places, loanHobby, verbTransitive}},
		// LOAN_DAILY は ADJ (コーヒー は おいしい)
		{"%s は %s", [][]string{loanDaily, adjectives}},
		// LOAN_PLACES は ADJ (ホテル は 高い)
		{"%s は %s", [][]string{loanPlaces, adjectives}},
		// TIME に LOAN_SURU する (朝 に ジョギング する)
		{"%s に %s する", [][]string{times, loanSuru}},
		// PERSON と LOAN_HOBBY を VERB_T (友達 と ゲーム を する)
		{"%s と %s を %s", [][]string{persons, loanHobby, verbTransitive}},
		// LOAN_TECH の LOAN_SURU する (パソコン の チェック する)
		{"%s の %s する", [][]string{loanTech, loanSuru}},
		// LOAN_DAILY を PERSON に VERB_T (コーヒー を お母さん に 作る)
		{"%s を %s に %s", [][]string{loanDaily, persons, verbTransitive}},
		// ADVERB LOAN_DAILY を VERB_T (すぐ コーヒー を 飲む)
		{"%s %s を %s", [][]string{adverbs, loanDaily, verbTransitive}},
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
