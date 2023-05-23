# What is it?
このリポジトリは[rinna/japanese-gpt-neox]([URL](https://huggingface.co/rinna))といったgpt-neoxベースのモデルを[ggml]([URL](https://github.com/ggerganov/ggml))形式のモデルに変換して使用することが前提の場合においてLoRAチューニングの成果物であるアダプタモデルをggmlモデルに適応して利用したい場合に必要なスクリプト改良とマージ用スクリプトを参考用・研究用に置いています。

# 注意事項
あくまでもこのリポジトリは参考用・研究用の目的で作成され、一般公開しています。作成者はジェネラリストである為、ディープな部分まで機械学習の専門家というわけではありませんし、必ずしもここに残されている記録が正しい結果や目的を達成できるとは限りません。

# 必須要件
## 前提環境
- [redpajama.cpp]([URL](https://github.com/togethercomputer/redpajama.cpp))
<br>
rinnnaといったgpt-neoxモデルをggmlに変換するのに必要な変換スクリプトが [/examples/redpajama/scripts](https://github.com/togethercomputer/redpajama.cpp/tree/master/examples/redpajama/scripts)に収められている
<br>
<br>
- [alpaca-lora](https://github.com/tloen/alpaca-lora)
<br>
知識があるなら1から準備した方が良いのは勿論だが、手軽にLoRAチューンをするならこれをベースにすると良さそう。gpt-neoxが使用できる前提の調整はされてはいないので改良必須。
<br>

# 作業前の大前提
- LoRAを当てずにオリジナルの状態で使用できるまでの手順や方法は事前にこなしておいて理解を深めておくべき。私の環境ではこちらの記事を参考にquantize前で止めてlangchainから呼び出せるようにしてあります。
[rinna 3Bをcppで動かす / by if001](https://note.com/if001/n/n6da85d0077d7)

# 主な改良概要・マージスクリプト情報
## ＋alpaca-lora/finetune.py
### 1:<br>
- Source
```
from transformers import LlamaForCausalLM, LlamaTokenizer
```
- EDITED
```
from transformers import AutoModelForCausalLM, AutoTokenizer
```

### 2:<br>
[peft@75808eb2a6e7b4c3ed8aec003b6eeb30a2db1495](https://github.com/huggingface/peft/commit/75808eb2a6e7b4c3ed8aec003b6eeb30a2db1495) のコミットで追加された記述の一部を削除しないと出力される"adapter_model.bin"がデータセットのパラメータ数に関係なく1kb前後という明らかに異常があるデータが保存されます。どうやらウェイト保存が正常に行われていないらしく2023/05/24時点で問題は解消していないようです。<br>
- 削除必須箇所について<br>
[huggingface/peft#286](https://github.com/huggingface/peft/issues/286#issuecomment-1501617281)

- 関連issue集<br>
[model.save_pretrained() produced a corrupted adapter_model.bin (only 443 B) with alpaca-lora
](https://github.com/huggingface/peft/issues/286)
<br>
<br>
[Maximum recursion depth exceeded](https://github.com/tloen/alpaca-lora/issues/37#issuecomment-1473140882)
<br>
<br>
[After fine-tuning, the model repeat the answer and not stop](https://github.com/tloen/alpaca-lora/issues/467)
<br>

## ＋alpaca-lora/merge_gptneox_lora.py
リポジトリ作成者であるYukaryavkaが参考コードを元に仕上げたマージ用スクリプト。重要なのは"L37～L43"のgpt-neox向けのレイヤーマージ処理部分

- 参考コード
[lvwerra/trl: examples/sentiment/scripts/gpt-neox-20b_peft/merge_peft_adapter.py](https://github.com/lvwerra/trl/blob/main/examples/sentiment/scripts/gpt-neox-20b_peft/merge_peft_adapter.py#L37)

# LoRAチューニング実行からマージまでのワークフロー
1. このリポジトリに置かれている"alpaca-lora"以下に配置されている2つのスクリプトを本家alpaca-loraへ配置・上書き
2. "rinna/japanese-gpt-neox-3.6b-instruction-sft"モデル向けのLoRAチューニング実行時のオプションを直にfinetune.pyへ書き込んである為、直接弄ってチューニングを実行するか"python3 finetune.py ～オプション"で各々の都合の良いように変更して使用してください。以下、弄っておいた方がいいオプション集
- data_path<br>
"./dataset.json"とダミー用の表記にしてあるので変更必須。
- output_dir<br>
"./rinna-3.6b-inst-lora"にしているので出力先を変更するならお忘れずに。
- batch_size & micro-batch-size<br>
128にした方が精度が良いらしい: [ソース](https://github.com/tloen/alpaca-lora/issues/191#issuecomment-1486275255)
<br>VRAM使用量を調整したいなら"micro-batch-size"を弄るとコントロール出来る。
- val_set_size<br>
データセットのパラメータ数が極端に少ないとデフォの2000ではエラーが出ることがあるのでデータセットのパラメータ数によっては調整する必要あり
3. output_dirにadapter_model.binとadapter_config.jsonが出力されている事を確認。ここでadapter_config.jsonの"base_model_name_or_path"を推論用のスクリプトを呼び出す位置を基準とした"ggmlに変換したモデルへの相対パス"へ差し替えておく。
4. "merge_gptneox_lora.py"を実行する事でLoRAアダプタとgptneoxベースモデルがマージされた新たなpytorch_model.binとconfigが出力されます。スクリプトの使用方法は別項にて
5. マージ済みのモデルが保存されているのでそれをredpajama.cppの [/examples/redpajama/scripts](https://github.com/togethercomputer/redpajama.cpp/tree/master/examples/redpajama/scripts)に置かれている変換用スクリプトで変換してfloat32 or float16の状態で読み込むもよし。quantize化して軽量化してから使用してもよし。後はご自由に。

# merge_gptneox_lora.py - リファレンス
準備中です... まだ"merge_gptneox_lora.py"をmainブランチへ加えていない為、準備ができるまでお待ち下さい。