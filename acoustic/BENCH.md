# acoustic ベンチマーク結果

実行環境: Apple M3 / darwin arm64

## 最新 (SIMD: ARM64 NEON Mahalanobis距離, WORDエンコーディング)

```
BenchmarkGMM_LogProb_1mix_39dim-8     	77835942	        15.49 ns/op	       0 B/op	       0 allocs/op
BenchmarkGMM_LogProb_4mix_39dim-8     	 6937311	       174.9 ns/op	       0 B/op	       0 allocs/op
BenchmarkGMM_LogProb_16mix_39dim-8    	 1378441	       934.8 ns/op	       0 B/op	       0 allocs/op
BenchmarkForward_100frames-8          	   18544	     65248 ns/op	    6784 B/op	       2 allocs/op
BenchmarkForward_500frames-8          	    3508	    346360 ns/op	   32768 B/op	       2 allocs/op
BenchmarkBaumWelch_10seq_50frames-8   	     681	   1774504 ns/op	   53142 B/op	     133 allocs/op
BenchmarkMahalanobisAccum_39dim-8     	69191347	        18.16 ns/op	       0 B/op	       0 allocs/op
```

注: GMM_16mixはSIMD関数呼び出しオーバーヘッド(ABI0, 1回~10ns)がコンポーネント数分累積するため退行。
実ワークロード(BaumWelch)ではSIMDによる個別Gaussian.LogProb高速化が効き、全体として改善。

## 前回 (emission cache, ワークスペース再利用, LogAdd閾値早期脱出, PrecomputeSoA修正)

```
BenchmarkGMM_LogProb_1mix_39dim-8     	44183322	        27.04 ns/op	       0 B/op	       0 allocs/op
BenchmarkGMM_LogProb_4mix_39dim-8     	 6995334	       169.3 ns/op	       0 B/op	       0 allocs/op
BenchmarkGMM_LogProb_16mix_39dim-8    	 1691246	       708.4 ns/op	       0 B/op	       0 allocs/op
BenchmarkForward_100frames-8          	   18309	     65339 ns/op	    6784 B/op	       2 allocs/op
BenchmarkForward_500frames-8          	    3728	    322031 ns/op	   32768 B/op	       2 allocs/op
BenchmarkBaumWelch_10seq_50frames-8   	     615	   1960731 ns/op	   53139 B/op	     133 allocs/op
```

## 初期実装

```
BenchmarkGMM_LogProb_1mix_39dim-8     	47961555	        24.96 ns/op	       0 B/op	       0 allocs/op
BenchmarkGMM_LogProb_4mix_39dim-8     	 5252682	       226.1 ns/op	       0 B/op	       0 allocs/op
BenchmarkGMM_LogProb_16mix_39dim-8    	 1250810	       960.4 ns/op	       0 B/op	       0 allocs/op
BenchmarkForward_100frames-8          	   15870	     75419 ns/op	    6784 B/op	       2 allocs/op
BenchmarkForward_500frames-8          	    3214	    374350 ns/op	   32768 B/op	       2 allocs/op
BenchmarkBaumWelch_10seq_50frames-8   	     262	   4605894 ns/op	  317850 B/op	     256 allocs/op
```

## 比較 (前回 → 最新)

| ベンチマーク | 前 ns/op | 後 ns/op | 変化 |
|---|---|---|---|
| GMM_1mix | 27.04 | 15.49 | **42.7%高速化** |
| GMM_4mix | 169.3 | 174.9 | ~同等 |
| GMM_16mix | 708.4 | 934.8 | 32.0%退行 (呼出OH) |
| Forward_100f | 65,339 | 65,248 | ~同等 |
| BaumWelch_10seq | 1,960,731 | 1,774,504 | **9.5%高速化** |

## 累積改善 (初期 → 最新)

| ベンチマーク | 初期 ns/op | 最新 ns/op | 速度改善 | 初期 B/op | 最新 B/op | メモリ削減 |
|---|---|---|---|---|---|---|
| GMM_1mix | 24.96 | 15.49 | **37.9%** | 0 | 0 | - |
| GMM_4mix | 226.1 | 174.9 | **22.6%** | 0 | 0 | - |
| Forward_100f | 75,419 | 65,248 | **13.5%** | 6,784 | 6,784 | 0% |
| Forward_500f | 374,350 | 346,360 | **7.5%** | 32,768 | 32,768 | 0% |
| BaumWelch_10seq | 4,605,894 | 1,774,504 | **61.5%** | 317,850 | 53,142 | **83.3%** |

最終更新: 2026-02-16
