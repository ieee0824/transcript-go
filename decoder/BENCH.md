# decoder ベンチマーク結果

実行環境: Apple M3 / darwin arm64

## 最新 (LogAdd閾値早期脱出適用)

```
BenchmarkDecode_5vocab_50frames-8     	      94	  12308694 ns/op	 1014340 B/op	    5895 allocs/op
BenchmarkDecode_10vocab_100frames-8   	      50	  23620384 ns/op	 2284157 B/op	    7742 allocs/op
BenchmarkDecode_20vocab_200frames-8   	      10	 106627542 ns/op	 4957156 B/op	   27453 allocs/op
```

## 前回 (トークンプーリング, wordHistoryリンクリスト化, スライス再利用)

```
BenchmarkDecode_5vocab_50frames-8     	      80	  13555136 ns/op	 1242545 B/op	    6560 allocs/op
BenchmarkDecode_10vocab_100frames-8   	      37	  32486229 ns/op	 2538185 B/op	   11843 allocs/op
BenchmarkDecode_20vocab_200frames-8   	      10	 101816258 ns/op	 4769929 B/op	   23553 allocs/op
```

## 初期実装

```
BenchmarkDecode_5vocab_50frames-8     	      78	  13813317 ns/op	10536045 B/op	  176648 allocs/op
BenchmarkDecode_10vocab_100frames-8   	      28	  41013051 ns/op	53218986 B/op	  545061 allocs/op
BenchmarkDecode_20vocab_200frames-8   	       6	 185771500 ns/op	363082344 B/op	 2081777 allocs/op
```

## 比較 (前回 → 最新)

| ベンチマーク | 前 ns/op | 後 ns/op | 速度改善 |
|---|---|---|---|
| 5vocab_50f | 13,555,136 | 12,308,694 | **9.2%** |
| 10vocab_100f | 32,486,229 | 23,620,384 | **27.3%** |

## 累積改善 (初期 → 最新)

| ベンチマーク | 初期 ns/op | 最新 ns/op | 速度改善 | 初期 B/op | 最新 B/op | メモリ削減 | 初期 allocs | 最新 allocs | allocs削減 |
|---|---|---|---|---|---|---|---|---|---|
| 5vocab_50f | 13,813,317 | 12,308,694 | **10.9%** | 10,536,045 | 1,014,340 | **90.4%** | 176,648 | 5,895 | **96.7%** |
| 10vocab_100f | 41,013,051 | 23,620,384 | **42.4%** | 53,218,986 | 2,284,157 | **95.7%** | 545,061 | 7,742 | **98.6%** |
| 20vocab_200f | 185,771,500 | 106,627,542 | **42.6%** | 363,082,344 | 4,957,156 | **98.6%** | 2,081,777 | 27,453 | **98.7%** |

最終更新: 2026-02-16
