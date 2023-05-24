# rinna_gpt-neox_ggml-lora

## Licence
- finetune.py - Apache License 2.0

[tloen/alpaca-lora](https://github.com/tloen/alpaca-lora)

元リポジトリより継承

- merge_gptneox_lora.py - Apache License 2.0

[Yukaryavka](https://github.com/Yukaryavka)

[lvwerra/trl: examples/sentiment/scripts/gpt-neox-20b_peft/merge_peft_adapter.py](https://github.com/lvwerra/trl/blob/main/examples/sentiment/scripts/gpt-neox-20b_peft/merge_peft_adapter.py#L37)

リポジトリ主が汎用利用可能な状態に仕上げましたが、レイヤーマージコード元のライセンスを継承しておきます。

## What is it?

このリポジトリは[rinna/japanese-gpt-neox]([URL](https://huggingface.co/rinna))といったgpt-neoxベースのモデルを[ggml]([URL](https://github.com/ggerganov/ggml))形式のモデルに変換して使用することが前提の場合においてLoRAチューニングの成果物であるアダプタモデルをggmlモデルに適応して利用したい場合に必要なスクリプト改良とマージ用スクリプトを参考用・研究用に置いています。

## 注意事項

あくまでもこのリポジトリは参考用・研究用の目的で作成され、一般公開しています。作成者はジェネラリストである為、ディープな部分まで機械学習の専門家というわけではありませんし、必ずしもここに残されている記録が正しい結果や目的を達成できるとは限りません。

## 必須要件

### ＞ 前提環境

- [redpajama.cpp](https://github.com/togethercomputer/redpajama.cpp)

rinnaといったgpt-neoxモデルをggmlに変換するのに必要な変換スクリプトが[/examples/redpajama/scripts](https://github.com/togethercomputer/redpajama.cpp/tree/master/examples/redpajama/scripts)に収められている

- [alpaca-lora](https://github.com/tloen/alpaca-lora)

知識があるなら1から準備した方が良いのは勿論だが、手軽にLoRAチューンをするならこれをベースにすると良さそう。gpt-neoxが使用できる前提の調整はされてはいないので改良必須。

### ＞ 作業前の大前提

LoRAを当てずにオリジナルの状態で使用できるまでの手順や方法は事前にこなしておいて理解を深めておくべき。私の環境ではこちらの記事を参考にquantize前で止めてlangchainから呼び出せるようにしてあります。<br>

[rinna 3Bをcppで動かす / by if001](https://note.com/if001/n/n6da85d0077d7)

## 主な改良概要・マージスクリプト情報

## ＋alpaca-lora/finetune.py

### **1:**

- **Source**

```python
from transformers import LlamaForCausalLM, LlamaTokenizer
```

- **EDITED**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
```

### **2:**

[peft@75808eb2a6e7b4c3ed8aec003b6eeb30a2db1495](https://github.com/huggingface/peft/commit/75808eb2a6e7b4c3ed8aec003b6eeb30a2db1495) のコミット辺りで追加された記述の一部を削除しないと出力される"adapter_model.bin"がデータセットのパラメータ数に関係なく1kb前後という明らかに異常があるデータが保存されます。どうやらウェイト保存が正常に行われていないらしく**2023/05/24時点で問題は解消していない**ようです。

- **削除必須箇所について**

[huggingface/peft#286](https://github.com/huggingface/peft/issues/286#issuecomment-1501617281)

- **関連issue集**

[model.save_pretrained() produced a corrupted adapter_model.bin (only 443 B) with alpaca-lora
](https://github.com/huggingface/peft/issues/286)

[Maximum recursion depth exceeded](https://github.com/tloen/alpaca-lora/issues/37#issuecomment-1473140882)

[After fine-tuning, the model repeat the answer and not stop](https://github.com/tloen/alpaca-lora/issues/467)

## ＋alpaca-lora/merge_gptneox_lora.py

リポジトリ作成者であるYukaryavkaが参考コードを元に仕上げたマージ用スクリプト。**重要なのは"L37～L43"のgpt-neox向けのレイヤーマージ処理部分**

- **参考コード**

[lvwerra/trl: examples/sentiment/scripts/gpt-neox-20b_peft/merge_peft_adapter.py](https://github.com/lvwerra/trl/blob/main/examples/sentiment/scripts/gpt-neox-20b_peft/merge_peft_adapter.py#L37)

レイヤーマージ処理部分のコードは上記のコードをそのまま使用するとエラーがL39で発生する為、以下の用に修正しfrom定義を追加する必要がある

- **Source**
  
```python
parent, target, target_name = model.base_model._get_submodules(key)
```

- **EDITED**

```python
parent, target, target_name = _get_submodules(model.base_model.model, key)
```

- **追加定義**

```python
from peft.utils import _get_submodules
```

- **上記修正に関連するissue**

[Cannot run stackllama example](https://github.com/lvwerra/trl/issues/287)

## LoRAチューニング実行からマージまでのワークフロー

1. このリポジトリに置かれている"alpaca-lora"以下に配置されている2つのスクリプトを本家alpaca-loraへ配置・上書き
2. "rinna/japanese-gpt-neox-3.6b-instruction-sft"モデル向けのLoRAチューニング実行時のオプションを直にfinetune.pyへ書き込んである為、直接弄ってチューニングを実行するか"python3 finetune.py ～オプション"で各々の都合の良いように変更して使用してください。以下、弄っておいた方がいいオプション集

- **data_path**
"./dataset.json"とダミー用の表記にしてあるので変更必須。
- **output_dir**
"./rinna-3.6b-inst-lora"にしているので出力先を変更するならお忘れずに。
- **batch_size & micro-batch-size**
128にした方が精度が良いらしい: [ソース](https://github.com/tloen/alpaca-lora/issues/191#issuecomment-1486275255)

   VRAM使用量を調整したいなら"micro-batch-size"を弄るとコントロール出来る。

- **val_set_size**
データセットのパラメータ数が極端に少ないとデフォの2000ではエラーが出ることがあるのでデータセットのパラメータ数によっては調整する必要あり

3. output_dirにadapter_model.binとadapter_config.jsonが出力されている事を確認。ここでadapter_config.jsonの"base_model_name_or_path"を推論用のスクリプトを呼び出す位置を基準とした"ggmlに変換したモデルへの相対パス"へ差し替えておく。
4. "merge_gptneox_lora.py"を実行する事でLoRAアダプタとgptneoxベースモデルがマージされた新たなpytorch_model.binとconfigが出力されます。スクリプトの使用方法は別項にて
5. マージ済みのモデルが保存されているのでそれをredpajama.cppの [/examples/redpajama/scripts](https://github.com/togethercomputer/redpajama.cpp/tree/master/examples/redpajama/scripts)に置かれている変換用スクリプトで変換してfloat32 or float16の状態で読み込むもよし。quantize化して軽量化してから使用してもよし。後はご自由に。

## merge_gptneox_lora.py - リファレンス

※問題報告やディスカッションを行いたい方は"issues"もしくは"discussions"をご利用ください。

- コマンドライン例

```cmd
python3 merge_gptneox_lora.py base_model_name lora_model_name output_dir
```

- **base_model_name**
LoRAをマージするベースモデル名・ディレクトリパス・huggingFaceリポジトリ名を定義する。 / 例: rinna/japanese-gpt-neox-3.6b-instruction-sft

- **lora_model_name**
ベースモデルにマージするLoRAモデルのディレクトリパスを定義する。(基本的には "adapter_model.bin" と "adapter_config.json" が格納されているディレクトリへのパスを設定する)

- **output_dir**
ベースモデルとLoRAモデルがマージされたpytorch_model.binとconfig.json郡を格納するディレクトリを定義する。

## 成果報告
※この成果報告はこちらの環境でテストした際の結果を報告しているだけです。確実にマージが成功しているかの判断材料としては、客観的意見が無いと信憑性に欠けることは明らかでしょう。このリポジトリを閲覧されている方で報告や分析を行って頂ける方はどうぞよろしくお願いします。

### ＞ **前提条件**
- データセットは
[OjousamaTalkScriptDataset](https://github.com/matsuvr/OjousamaTalkScriptDataset)の200プロンプト+追加55プロンプトに
"instruction"と"output"の出力に私の架空のキャラクター「Sizryavka」の人格や喋るセリフ設定に合わせたものに全て改変したカスタムデータセットを使用しています。

※どんなキャラ(口調)か知りたい人は以下のキャラクター目線で仕上げたキャラクターロールプレイ日誌・小説「FallsFrontline」を閲覧してください。

[【Fallout4 x ドルフロ】Falls Frontline - Sizryavkaの終末世界活動ログ / Log_No.1](https://yukaryavka.tumblr.com/post/704896045069598720/fallout4-x-%E3%83%89%E3%83%AB%E3%83%95%E3%83%ADfalls-frontline)
- **epoch: 3** / 当リポジトリの"merge_gptneox_lora.py"を使用して**LoRAモデルマージ済みのモデルをggml/f32の状態で推論。**

### ＞ **結果**
- [画像付き報告用ツイート1: データセットに書かれているinstructionをユーザー入力として渡した結果](https://twitter.com/Yukaryavka/status/1661431139603738624)
- [画像付き報告用ツイート2: データセットのinstructionには存在していないユーザー入力を渡した結果](https://twitter.com/Yukaryavka/status/1661442123902877697)

### ＞ **あとがき**
データセット数がそもそも少なすぎるのもあり、突然素のような結果を返すパターンもそもそも支離滅裂な文章を返すパターンも勿論ありました。ただ、2～3回の生成のうち1回は必ずそれっぽい文章は出力するようになってはいるように思えます。
もう少しLoRAモデルが強調されるように色々調整すれば進展はありそうです。ただ、これらの情報を挙げた所で直接マージ済みモデルのレイヤー精査やモデル評価といった分析に詳しい方による調査は必要だと思われます。(分析方面は疎いものでして自己調査はちょっと厳しいですね。)
後は大規模なデータセットによるLoRAチューニングモデルでの同様の調査を行うべきですが、他にもやる事があるのと手持ちの演算リソースが乏しいので私の方での報告や基本的な調査はここまでにしておきます。後は各々で試していただいて、ここに載っている使える情報だけ使っていただければ。
