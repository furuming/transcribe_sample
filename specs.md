### 要件
- Faster-Whisperのtinyを用いて音声文字起こしを行う
- 実行はrun.pyをコマンドラインから呼び出す
- 音声ファイルは実行時の引数で指定
- 出力はwhisperの実行結果をjson化
- 出力はoutput/音声ファイル名_実行日時とする
- ライブラリのインストールはuvを使う
- .envにHF_TOKENがあればHuggingFaceにログインする

- 今回はすべてCPUで演算する
