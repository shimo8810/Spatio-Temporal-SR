# STSR(Spatio-Temporal Super-Resolution)
時空間超解像の研究用repository

## ファイル構成
```
Spatio-Temporal-SR/
    ├ experiments/
    ├ results/
    ├ libs/
    ├ scripts/
    ├ datasets/
    ├ Pubs/ # Dropboxへのsymbolic link
        ├ Docs/
        ├ Slides/
        ├ MTs/
```
実験系ディレクトリ
- experiments
実験コードとかが入っている.
- results
experimentsの実験に対応した結果が入っている.
- experiments
共通で使うlib.

- scripts
共通のスクリプト, データセット生成とか
- datasets
データセットディレクトリ, symbolic link. 開発マシンなら極小テストデータディレクトリ, サーバなら普通のデータセットがあるディレクトリにリンクされている.

その他ディレクトリ
- Pubs
公開物(Publications)ディレクトリ, ドキュメントやスライド, MTで使う資料が入っているが, 基本的にはDropBoxで管理.
Docsは論文, slideはスライド. MTsは先生とのMT用.
