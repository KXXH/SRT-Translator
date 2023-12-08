from openai import OpenAI
import srt
import itertools
import time
import re
import argparse


SYSTEM_PROMPT = """假设你是一名专业的翻译人员，请你将一段字幕翻译为中文。
要求：
1. 由于是字幕，因此在需要翻译的文字中可能会出现一些意料之外的换行。这些换行是为了保证字幕能正确地在屏幕上显示，你需要原样保留这些换行。
2. 由于换行完全是出于格式目的，因此换行不代表一个句子的结束。无论如何，一个句子必定以“{”开始，以“}”结束。
3. 我们使用“{“和”}”来表示一个句子的开始和结束。无论如何，一个句子必定以“{”开始，以“}”结束。
4. 你是一名专业的翻译，因此你不能根据语气来自作主张地添加或删除大括号。你必须要保证每一个句子都被正确地大括号包裹。
5. 遇到人名和地名无需翻译。
6. 我们会给你小费，但是如果你的翻译不符合要求，我们会扣除小费。
例子：
输入：
{I'm a translator.}{I'm a translator.}{Hello World}
输出：
{我是一名翻译人员。}{我是一名翻译人员。}{你好世界}
下面，是你需要翻译的内容：\n\n"""


def translate_srt(srt_data, api_key, **kwargs):
    openai = OpenAI(api_key=api_key, base_url=kwargs.get('base_url'))
    model = kwargs.get('model', 'gpt-3.5-turbo')
    window_size = kwargs.get('window_size', 10)
    step_size = kwargs.get('step_size', 10)
    wait_sec = kwargs.get('wait_sec', 1)

    system_prompt = kwargs.get('system_prompt', SYSTEM_PROMPT)
    i = 0
    translated_srt = []
    while i < len(srt_data):
        srt_to_translate = list(srt_data[i:i+window_size])
        prompt = system_prompt + \
            "".join(
                map(lambda x: "{" + re.sub(r'{(.*?)}', '', x.content) + "}", srt_to_translate))
        res = openai.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=model
        )
        translate_raw_content = res.choices[0].message.content
        splited_content = re.findall(
            r'{(.*?)}', translate_raw_content, re.DOTALL)
        if len(splited_content) != len(srt_to_translate):
            print(
                f"splited_content length({len(splited_content)} does not equal to srt_to_translate length {len(srt_to_translate)}, retry...")
            time.sleep(wait_sec)
            continue
        for origin, translate in zip(srt_to_translate, splited_content):
            new_srt = srt.Subtitle(
                index=origin.index, start=origin.start, end=origin.end, content=translate)
            translated_srt.append(new_srt)
        i += step_size
        time.sleep(wait_sec)
    return translated_srt


if __name__ == "__main__":
    # 创建解析器
    parser = argparse.ArgumentParser(description="Translate SRT files.")

    # 添加参数
    parser.add_argument("--api_key", required=True, help="API key for OpenAI.")
    parser.add_argument("--base_url", required=True,
                        help="Base URL for OpenAI.")
    parser.add_argument("--srt_path", required=True,
                        help="Path to the SRT file to be translated.")
    parser.add_argument("--window_size", default=10, type=int,
                        help="Window size for translation.")
    parser.add_argument("--step_size", default=10, type=int,
                        help="Step size for translation.")
    parser.add_argument("--output_path", required=True,
                        help="Path to the output translated SRT file.")

    # 解析参数
    args = parser.parse_args()

    # 使用参数
    openai = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url
    )

    SRT_PATH = args.srt_path
    with open(SRT_PATH, 'r') as f:
        srt_data = list(srt.parse(f.read()))

    WINDOW_SIZE = args.window_size
    STEP_SIZE = args.step_size

    translated_srt = translate_srt(
        srt_data, args.api_key, window_size=WINDOW_SIZE, step_size=STEP_SIZE)

    OUTPUT_PATH = args.output_path
    with open(OUTPUT_PATH, 'w') as f:
        f.write(srt.compose(translated_srt))
