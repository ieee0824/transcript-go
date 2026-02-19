# transcript

Go言語によるフルスクラッチ日本語音声認識ライブラリ。
HMM-GMM音響モデルとN-gram言語モデルによる古典的な音声認識パイプラインを純粋なGoで実装しています。

## 特徴

- **純粋Go実装** — クロスプラットフォーム対応 (macOSではApple Accelerate自動活用)
- **トライフォン対応HMM-GMM音響モデル** — 日本語29音素、5状態left-to-right HMM、対角共分散GMM、文脈依存トライフォンHMMによる高精度認識
- **N-gram言語モデル** — ARPA形式の読み込み、Witten-Bellスムージング付きbigram/trigram対応、OOV処理
- **MFCC特徴量抽出** — 39次元 (13 MFCC + 13 Δ + 13 ΔΔ)、ケプストラム平均正規化 (CMN)、VTLN (声道長正規化)、FFT NEON/SSE2アセンブリ + マルチコア並列化
- **レキシコンプレフィックスツリーデコーダ** — LMルックアヘッド、トライグラム再結合、発射キャッシュによる高速ビームサーチ
- **Baum-Welch (EM) 訓練** — 強制アラインメント、モノフォン→トライフォン段階訓練、goroutine並列化
- **言語モデルビルダー** — コーパスからWitten-Bellスムージング付きARPA形式を自動生成
- **自然言語テキストフィルタ** — MeCab + 辞書照合でWikipedia等からLM学習用コーパスを自動生成

## 学習済みモデル

`models/v11/` に学習済みモデルが同梱されています。

| ファイル | 内容 |
|---|---|
| `models/v11/am.gob` | 音響モデル (55話者TTS, 5,794発話, 5-way speed augment, 4-mix GMM, トライフォン) |
| `models/v11/lm.arpa` | 言語モデル (トライグラム, テンプレート14,250文 + Wikipedia抽出393文) |
| `models/v11/dict.txt` | 発音辞書 (1,176語) |

### クイックスタート

```bash
go build -o /tmp/transcript ./cmd/transcript/

/tmp/transcript \
    -am models/v11/am.gob \
    -lm models/v11/lm.arpa \
    -dict models/v11/dict.txt \
    -wav input.wav \
    -oov-prob -5.0 -lm-weight 10.0 -max-tokens 5000
```

## 認識精度

### v11 (最新)

| テスト条件 | 精度 |
|---|---|
| コーパス内文 × 外部話者 | 90% |
| 感情音声 × 外部話者 | 90% |
| 特定TTS話者 | 50% |
| コーパス外文 × 未知TTS話者 (3話者平均) | 57% |

### v10

| テスト条件 | 精度 |
|---|---|
| 学習データ (2,469発話 × 5-way augment, TTS 30話者) | 98.4% |
| コーパス内文 × 外部話者 | 90% |
| 感情音声 × 外部話者 | 90% |
| コーパス外文 × 未知TTS話者 (3話者平均) | 47% |

## プロジェクト構成

```
transcript/
├── transcript.go          トップレベルAPI (Recognizer)
├── models/v11/            学習済みモデル (AM + LM + 辞書)
├── cmd/
│   ├── transcript/        音声認識CLI
│   ├── train/             音響モデル訓練CLI
│   ├── lmbuild/           言語モデルビルダー
│   ├── lmtext/            自然言語テキストフィルタ (MeCab + 辞書照合)
│   ├── wikitext/          MediaWiki XMLダンプからテキスト抽出
│   ├── cvimport/          Common Voice日本語コーパスインポート
│   ├── corpusgen/         テンプレートベースコーパス生成
│   ├── dictconv/          辞書変換 (MeCab辞書 → 発音辞書)
│   └── dictfilter/        辞書フィルタリング
├── acoustic/              音響モデル (HMM + GMM, トライフォン, Baum-Welch訓練)
├── audio/                 WAVファイル読み込み (16bit PCM, mono, 16kHz)
├── feature/               MFCC特徴量抽出 (FFT, Melフィルタバンク, DCT, デルタ, CMN)
├── language/              言語モデル (N-gram, ARPA形式パーサ, ビルダー)
├── lexicon/               発音辞書 (単語 → 音素列)
├── decoder/               レキシコンプレフィックスツリーデコーダ
├── internal/blas/         Apple Accelerate cblas_dgemm wrapper + 純Goフォールバック
├── internal/simd/         NEON (arm64) / SSE2 (amd64) SIMDアセンブリ
├── internal/mathutil/     ログ域算術、ベクトル/行列演算
└── training/              訓練用コーパス
```

## 必要条件

- Go 1.24 以上

## ビルド

```bash
go build -o /tmp/transcript ./cmd/transcript/
```

## テスト

```bash
go test ./... -timeout 60s
```

## CLI の使い方

### 音声認識

```bash
transcript -am models/v11/am.gob -lm models/v11/lm.arpa -dict models/v11/dict.txt -wav input.wav
```

| フラグ | デフォルト | 説明 |
|---|---|---|
| `-am` | (必須) | 音響モデルファイルのパス |
| `-lm` | (必須) | 言語モデルファイルのパス (ARPA形式) |
| `-dict` | (必須) | 発音辞書ファイルのパス |
| `-wav` | (必須) | 入力WAVファイルのパス (16bit PCM, mono, 16kHz) |
| `-beam` | 200.0 | ビーム幅 |
| `-lm-weight` | 10.0 | 言語モデルの重み |
| `-word-penalty` | 0.0 | 単語挿入ペナルティ |
| `-max-tokens` | 1000 | 最大アクティブトークン数 |
| `-oov-prob` | 0 | OOV語のunigram log10確率 (例: -5.0) |
| `-lm-interp` | 0.0 | LM補間重み (0=純LM, 0.5=半均一) |
| `-vtln` | false | VTLN話者正規化を有効化 (αグリッドサーチ) |
| `-v` | false | 詳細出力 (スコア、単語タイミング) |

### 音響モデル訓練

```bash
go run ./cmd/train -manifest data/manifest.tsv -dict data/dict.txt \
    -output model.gob -mix 4 -iter 20 -align-iter 2 \
    -triphone -augment
```

| フラグ | デフォルト | 説明 |
|---|---|---|
| `-manifest` | data/training/manifest.tsv | 訓練マニフェスト (wav_path\<TAB\>words) |
| `-dict` | data/dict.txt | 発音辞書 |
| `-output` | data/am.gob | 出力モデルパス |
| `-mix` | 1 | GMMコンポーネント数 |
| `-iter` | 20 | Baum-Welchイテレーション数 |
| `-align-iter` | 0 | 強制アラインメント再訓練数 |
| `-triphone` | false | トライフォン訓練を有効化 |
| `-augment` | false | 5-way速度変換データ拡張 |
| `-manifest-noaug` | "" | 追加マニフェスト (速度拡張なし) |
| `-min-tri-seg` | 10 | トライフォンHMMの最小セグメント数 |

### 言語モデル構築

```bash
# バイグラム
go run ./cmd/lmbuild -output lm.arpa corpus1.txt corpus2.txt

# トライグラム
go run ./cmd/lmbuild -order 3 -output lm.arpa corpus1.txt corpus2.txt
```

### 自然言語テキストフィルタ

Wikipedia等の自然言語テキストから辞書収録語のみで構成される文を抽出し、LM学習用コーパスを生成します。MeCabが必要です。

```bash
# MediaWiki XMLダンプからテキスト抽出
go run ./cmd/wikitext jawiki-latest-pages-articles.xml.bz2 > wiki_sentences.txt

# 辞書フィルタリング
go run ./cmd/lmtext -dict models/v11/dict.txt < wiki_sentences.txt > nlp_corpus.txt

# テンプレートコーパスと統合してトライグラムLM構築
go run ./cmd/lmbuild -order 3 -output lm.arpa training/corpus8_expanded.txt nlp_corpus.txt
```

### Common Voiceコーパスインポート

Mozilla Common Voice日本語コーパスから辞書収録語のみで構成される発話を抽出し、MP3→WAV変換してマニフェストを生成します。MeCabとffmpegが必要です。

```bash
go run ./cmd/cvimport \
  -cv-dir /path/to/cv-corpus/ja \
  -dict models/v11/dict.txt \
  -output data/cv_manifest.tsv \
  -wav-dir data/cv_wav \
  -min-words 3 -min-votes 2
```

## ライブラリとしての使い方

### ファイルから認識

```go
rec, err := transcript.NewRecognizer("models/v11/am.gob", "models/v11/lm.arpa", "models/v11/dict.txt",
    transcript.WithDecoderConfig(decoder.Config{
        BeamWidth:       200.0,
        LMWeight:        10.0,
        MaxActiveTokens: 5000,
    }),
    transcript.WithOOVLogProb(-5.0),
)
if err != nil {
    log.Fatal(err)
}

result, err := rec.RecognizeFile("input.wav")
if err != nil {
    log.Fatal(err)
}
fmt.Println(result.Text)
```

### モデルを直接渡して認識

```go
rec := transcript.NewRecognizerFromModels(am, lm, dict,
    transcript.WithDecoderConfig(decoder.Config{
        BeamWidth:   300.0,
        LMWeight:    10.0,
    }),
)

result, err := rec.RecognizeSamples(samples)
```

## 入力ファイル形式

### WAV
- フォーマット: PCM (非圧縮)
- サンプリングレート: 16,000 Hz
- ビット深度: 16bit
- チャンネル: モノラル

### 発音辞書 (TSV)

```
東京	トウキョウ	t o u k y o u
食べる	タベル	t a b e r u
```

### 言語モデル (ARPA形式)

```
\data\
ngram 1=3
ngram 2=2

\1-grams:
-1.0  </s>
-1.0  <s>   -0.5
-0.5  東京

\2-grams:
-0.3  <s>  東京

\end\
```

## アーキテクチャ

### 音響モデル

- **モノフォンHMM**: 29音素 × 5状態 (入口・出口 + 3発射状態) のleft-to-right HMM
- **トライフォンHMM**: `left-center+right` 形式の文脈依存モデル。単語境界は `#` で表現 (例: `#-i+k`, `i-k+u`, `k-u+#`)
- **GMM**: 対角共分散ガウス混合モデル。各発射状態にMコンポーネント。バッチ行列積 (macOS: Accelerate/AMX) による高速尤度計算
- **ResolveHMM**: トライフォンHMM → モノフォンHMMへのフォールバック解決
- **訓練**: Baum-Welch → 強制アラインメント → トライフォン分割の段階的訓練

### デコーダ

- **レキシコンプレフィックスツリー**: 辞書全体をトライ構造に展開。トライフォン文脈でノード分岐
- **LMルックアヘッド**: ツリー内の各ノードで到達可能な最良LMスコアを事前計算
- **トライグラム再結合**: `(nodeIdx, stateIdx, lastWord, prevWord)` をキーにした重複トークンの統合
- **発射キャッシュ**: HMMポインタ同一性に基づくGMM計算結果のキャッシュ

### 特徴量

- 39次元MFCC (13 MFCC + 13 Δ + 13 ΔΔ)
- プリエンファシス → ハミング窓 → FFT → Melフィルタバンク → DCT → デルタ
- ケプストラム平均正規化 (CMN) による話者/チャンネル正規化
- VTLN (声道長正規化): 区分線形周波数ワーピングによるαグリッドサーチ (α ∈ 0.82–1.20)
- **FFT SIMDアセンブリ**: split R/Iレイアウト + NEON (arm64) / SSE2 (amd64) バタフライ演算
- **マルチコア並列化**: `runtime.NumCPU` goroutineによるフレーム並列処理 (30秒音声で3.6倍高速化)

## 日本語音素セット (29音素)

| カテゴリ | 音素 |
|---|---|
| 無音・ポーズ | sil, sp |
| 母音 | a, i, u, e, o |
| 破裂音 | k, g, t, d, p, b |
| 摩擦音 | s, z, h, f |
| 破擦音 | ch, ts, j |
| 鼻音 | m, n, ng |
| 流音 | r |
| 半母音 | y, w |
| 歯擦音 | sh |
| 特殊 | q (促音っ), long (長音ー) |

## 訓練データの準備

### マニフェスト (TSV)

```
/path/to/0000.wav	東京 タワー に 行く
/path/to/0001.wav	今日 は いい 天気 です
```

### コーパス (言語モデル用)

1行1文、スペース区切り:

```
東京 タワー に 行く
魚 を 焼く
友達 と 遊ぶ
```

### 訓練パイプライン

```bash
# 1. 発音辞書の生成 (MeCab辞書から)
go run ./cmd/dictconv -mecab /path/to/mecab-dict -output data/dict.txt

# 2. 辞書フィルタリング (オプション)
go run ./cmd/dictfilter data/dict.txt data/smalldict.txt 4000 > data/dict_filtered.txt

# 3. コーパス生成
go run ./cmd/corpusgen > training/corpus.txt

# 4. 音響モデル訓練 (トライフォン)
go run ./cmd/train -manifest data/manifest.tsv -dict data/dict.txt \
    -output data/am.gob -mix 4 -iter 20 -align-iter 2 \
    -triphone -augment

# 5. 言語モデル構築 (トライグラム)
go run ./cmd/lmbuild -order 3 -output data/lm.arpa training/corpus.txt
```
