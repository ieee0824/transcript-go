# transcript

Go言語によるフルスクラッチ日本語音声認識ライブラリ。
HMM-GMM音響モデルとn-gram言語モデルによる古典的な音声認識パイプラインを純粋なGoで実装しています。

## 特徴

- **純粋Go実装** — CGo依存なし、クロスプラットフォーム対応
- **HMM-GMM音響モデル** — 日本語29音素、5状態left-to-right HMM、対角共分散GMM
- **n-gram言語モデル** — ARPA形式の読み込み、バックオフ付きbigram/trigram対応
- **MFCC特徴量抽出** — 39次元 (13 MFCC + 13 Δ + 13 ΔΔ)
- **Viterbiビームサーチデコーダ**
- **Baum-Welch (EM) 訓練**

## プロジェクト構成

```
transcript/
├── cmd/transcript/        CLI
├── internal/mathutil/     ログ域算術、ベクトル/行列演算
├── audio/                 WAVファイル読み込み (16bit PCM, mono, 16kHz)
├── feature/               MFCC特徴量抽出 (FFT, Melフィルタバンク, DCT, デルタ)
├── acoustic/              音響モデル (HMM + GMM, Baum-Welch訓練, シリアライズ)
├── language/              言語モデル (n-gram, ARPA形式パーサ)
├── lexicon/               発音辞書 (単語 → 音素列)
├── decoder/               Viterbiビームサーチデコーダ
└── transcript.go          トップレベルAPI (Recognizer)
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

```bash
transcript -am model.gob -lm lm.arpa -dict dict.txt -wav input.wav
```

### オプション

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
| `-v` | false | 詳細出力 (スコア、単語タイミング) |

## ライブラリとしての使い方

### ファイルから認識

```go
rec, err := transcript.NewRecognizer("model.gob", "lm.arpa", "dict.txt")
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
        LMWeight:    15.0,
    }),
)

result, err := rec.RecognizeSamples(samples)
```

### 音響モデルの訓練

```go
// HMMの作成と訓練
hmm := acoustic.NewPhonemeHMM(acoustic.PhonA, 39, 4)
err := acoustic.TrainPhoneme(hmm, trainingSequences, acoustic.DefaultTrainingConfig())

// モデルの保存
am := acoustic.NewAcousticModel(39, 4)
am.Phonemes[acoustic.PhonA] = hmm
am.Save(file)
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
