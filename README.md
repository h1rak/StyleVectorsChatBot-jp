# StyleVectorsChatBot-jp

WRIMEデータセットとStyle Vectorsを使用した日本語感情制御チャットボットプロジェクト

## セットアップ

### 1. フォルダ構成の準備
```bash
# 必要なフォルダを作成
mkdir -p data
mkdir -p models
mkdir -p output/activations
```

### 2. データセットの準備
```bash
# dataフォルダを作成し、WRIMEデータをダウンロード
cd data
# WRIMEデータセットをダウンロード（公式リポジトリから）
# 注意: CC BY-NC-ND 4.0ライセンスに従い、改変禁止・非商用のみ
```

### 3. モデルの準備
```bash
# modelsフォルダを作成し、Qwen3-32Bをダウンロード
cd models
# Qwen3-32Bモデルをダウンロード
# 注意: モデルの利用規約を確認してください
```

### 4. 実行手順

#### 学習（隠れ層アクティベーション抽出）
```bash
python scripts/training/get_hidden_activations.py
```

#### 実行（スタイル制御生成）
```bash
python scripts/generation/steering.py
```

## ライセンスと使用制限

### WRIMEデータセット
- **ライセンス**: CC BY-NC-ND 4.0
- **制限事項**: 
  - ✅ **非商用利用のみ**: 研究・教育目的に限定
  - ❌ **改変禁止**: データセットの改変・加工は禁止
  - ✅ **表示義務**: 適切なクレジットとライセンス表示が必要
  - ❌ **商用利用禁止**: いかなる商用目的での使用も禁止

### Style Vectorsコード
- **出典**: Konen et al. (2024) "Style Vectors for Steering Generative Large Language Models"
- **リポジトリ**: https://github.com/DLR-SC/style-vectors-for-steering-llms
- **注意**: 元リポジトリのライセンスを確認してください

### Qwen3-32Bモデル
- **注意**: モデルの利用規約を確認し、遵守してください
- **商用利用**: モデルの利用規約に従ってください

## 使用上の重要な注意事項

### 🚫 禁止事項
- **商用利用**: このプロジェクト全体の商用利用は禁止
- **WRIMEデータの改変**: データセットの改変・加工・前処理結果の公開
- **WRIMEデータの再配布**: 元のデータセットの直接配布
- **営利目的での使用**: 直接的・間接的な営利活動での使用

### ✅ 許可される使用
- **学術研究**: 大学や研究機関での研究活動
- **教育目的**: 授業や演習での使用
- **個人学習**: 非商用の個人的な学習・研究
- **オープンソース貢献**: 非商用のオープンソースプロジェクト

### Style Vectors
```bibtex
@inproceedings{konen-etal-2024-style,
    title = "Style Vectors for Steering Generative Large Language Models",
    author = "Konen, Kai and Jentzsch, Sophie and Diallo, Diaoulé and Schütt, Peer and Bensch, Oliver and El Baff, Roxanne and Opitz, Dominik and Hecking, Tobias",
    booktitle = "Findings of the Association for Computational Linguistics: EACL 2024",
    year = "2024",
    pages = "782--802",
}
```

## データセット取得方法

### WRIMEデータセット
1. **公式リポジトリ**: https://github.com/ids-cv/wrime
2. **HuggingFace**: https://huggingface.co/datasets/shunk031/wrime
3. **注意**: ライセンス条項を確認し、非商用利用であることを確認してください

