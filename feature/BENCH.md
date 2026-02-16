# feature ベンチマーク結果

実行環境: Apple M3 / darwin arm64

## 最新 (最適化後: Melフィルタバンクスパース化, LogAdd閾値早期脱出)

```
BenchmarkFFT_512-8               	  188688	      6309 ns/op	    8192 B/op	       1 allocs/op
BenchmarkPowerSpectrum_512-8     	  174502	      6869 ns/op	   18688 B/op	       3 allocs/op
BenchmarkMelFilterbank_Apply-8   	 3299158	       369.3 ns/op	     208 B/op	       1 allocs/op
BenchmarkExtract_1sec-8          	    1302	    935144 ns/op	  602031 B/op	     573 allocs/op
BenchmarkExtract_5sec-8          	     255	   4683265 ns/op	 2708561 B/op	    2573 allocs/op
BenchmarkExtract_30sec-8         	      40	  28399372 ns/op	15850624 B/op	   15073 allocs/op
```

## 前回 (FFTワークスペース再利用, DCT/リフタテーブル事前計算)

```
BenchmarkFFT_512-8               	  174932	      6818 ns/op	    8192 B/op	       1 allocs/op
BenchmarkPowerSpectrum_512-8     	  161174	      7448 ns/op	   18688 B/op	       3 allocs/op
BenchmarkMelFilterbank_Apply-8   	  164529	      7474 ns/op	     208 B/op	       1 allocs/op
BenchmarkExtract_1sec-8          	     709	   1688302 ns/op	  597309 B/op	     546 allocs/op
BenchmarkExtract_5sec-8          	     139	   8561370 ns/op	 2703826 B/op	    2546 allocs/op
BenchmarkExtract_30sec-8         	      22	  51125383 ns/op	15845901 B/op	   15046 allocs/op
```

## 初期実装

```
BenchmarkFFT_512-8               	  158072	      6651 ns/op	    8192 B/op	       1 allocs/op
BenchmarkPowerSpectrum_512-8     	  149901	      8030 ns/op	   18688 B/op	       3 allocs/op
BenchmarkMelFilterbank_Apply-8   	  169899	      6998 ns/op	     208 B/op	       1 allocs/op
BenchmarkExtract_1sec-8          	     625	   1890241 ns/op	 2435253 B/op	     918 allocs/op
BenchmarkExtract_5sec-8          	     123	   9691024 ns/op	12100269 B/op	    4519 allocs/op
BenchmarkExtract_30sec-8         	      20	  57341412 ns/op	72482305 B/op	   27019 allocs/op
```

## 比較 (前回 → 最新)

| ベンチマーク | 前 ns/op | 後 ns/op | 速度改善 |
|---|---|---|---|
| MelFilterbank_Apply | 7,474 | 369.3 | **95.1%** (20.2x) |
| Extract_1sec | 1,688,302 | 935,144 | **44.6%** |
| Extract_5sec | 8,561,370 | 4,683,265 | **45.3%** |
| Extract_30sec | 51,125,383 | 28,399,372 | **44.4%** |

## 累積改善 (初期 → 最新)

| ベンチマーク | 初期 ns/op | 最新 ns/op | 速度改善 | 初期 B/op | 最新 B/op | メモリ削減 |
|---|---|---|---|---|---|---|
| MelFilterbank_Apply | 6,998 | 369.3 | **94.7%** (18.9x) | 208 | 208 | 0% |
| Extract_1sec | 1,890,241 | 935,144 | **50.5%** | 2,435,253 | 602,031 | **75.3%** |
| Extract_5sec | 9,691,024 | 4,683,265 | **51.7%** | 12,100,269 | 2,708,561 | **77.6%** |
| Extract_30sec | 57,341,412 | 28,399,372 | **50.5%** | 72,482,305 | 15,850,624 | **78.1%** |

最終更新: 2026-02-16
