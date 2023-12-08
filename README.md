# README

本 README 由 ChatGPT 生成。

## 项目介绍

这是一个使用 OpenAI API 进行 SRT 字幕文件翻译的 Python 项目。它可以读取 SRT 字幕文件，将其中的文本翻译成另一种语言，然后将翻译后的字幕写入到新的 SRT 文件中。

## 使用方法

首先，你需要安装项目的依赖。你可以通过运行以下命令来安装：

```sh
pip install -r requirements.txt
```

然后，你可以通过以下命令来运行项目：

```python
python main.py --api_key YOUR_OPENAI_API_KEY --base_url YOUR_OPENAI_BASE_URL --srt_path PATH_TO_YOUR_SRT_FILE --window_size WINDOW_SIZE --output_path PATH_TO_OUTPUT_FILE
```

其中：

- `YOUR_OPENAI_API_KEY` 是你的 OpenAI API 密钥。
- `YOUR_OPENAI_BASE_URL` 是你的 OpenAI API 的基础 URL。
- `PATH_TO_YOUR_SRT_FILE` 是你要翻译的 SRT 字幕文件的路径。
- `WINDOW_SIZE` 是翻译窗口的大小。翻译窗口越大，翻译效果越好，但单次翻译越容易出错，导致重试次数增加，增加翻译时间和 token 消耗。默认为 10，即每次翻译 10 行字幕。
- `PATH_TO_OUTPUT_FILE` 是输出翻译后的 SRT 文件的路径。

## 注意事项

- 请确保你的 OpenAI API 密钥和基础 URL 是正确的，否则项目无法正常运行。
- 请确保你的 SRT 字幕文件是有效的，否则项目无法正常运行。
- `WINDOW_SIZE` 的值会影响翻译的效果，你可以根据需要进行调整。
- 输出的 SRT 文件会覆盖同名的已存在文件，请确保你的输出路径是正确的。
