from openai import OpenAI, AsyncOpenAI
import srt
import asyncio
import time
import re
import argparse


SYSTEM_PROMPT = """我需要你帮忙翻译一些SRT字幕内容。下面是一些具体的要求：

1. 每一条字幕都已经被大括号（{}）包裹，例如：{Hello, world!}。你需要在大括号内进行翻译，翻译后的内容也要放在相同的大括号中。
2. 如果字幕中有换行，也请保留原样。
3. 如果你遇到人名或地名，请直接保留，不要进行翻译。
作为示例：如果原文字幕{Hello, world!\nMy name is John Doe.}需要从英语翻译为法语，结果应该是{Bonjour, le monde!\nMon nom est John Doe.}

原语言：<lang_src>
目标语言：<lang_tgt>

需要你翻译的字幕如下：

<content>

现在，补完下列翻译:

<example>
"""


async def async_request(openai, prompt, model, semaphore):
    async with semaphore:
        res = await openai.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=model
        )
        return res.choices[0].message.content


def translate_srt(srt_data, api_key, **kwargs):
    # init params
    openai = OpenAI(api_key=api_key, base_url=kwargs.get(
        'base_url'), timeout=30, max_retries=3)
    # openai = AsyncOpenAI(api_key=api_key, base_url=kwargs.get('base_url'))
    model = kwargs.get('model', 'gpt-3.5-turbo')
    window_size = kwargs.get('window_size', 10)
    step_size = kwargs.get('step_size', 10)
    wait_sec = kwargs.get('wait_sec', 1)
    lang_src = kwargs.get('lang_src', 'en')
    lang_tgt = kwargs.get('lang_tgt', 'zh')
    debug = kwargs.get('debug', False)
    padding = kwargs.get('padding', 0)
    max_tasks = asyncio.Semaphore(kwargs.get('max_tasks', 10))
    system_prompt = kwargs.get('system_prompt', SYSTEM_PROMPT)
    i = 0
    translated_srt = []

    while i < len(srt_data):
        has_front_padding = padding > 0 and i - padding > 0
        has_back_padding = padding > 0 and i + \
            WINDOW_SIZE + padding < len(srt_data)
        start = max(0, i - padding) if has_front_padding else i
        end = min(len(srt_data), i + WINDOW_SIZE +
                  padding) if has_back_padding else i + WINDOW_SIZE
        srt_to_translate = list(srt_data[start:end])
        example_translate_srt = list(
            translated_srt[start:i])  # [i - PADDING:i] or []

        # build prompt and send request
        prompt = system_prompt\
            .replace("<lang_src>", lang_src)\
            .replace("<lang_tgt>", lang_tgt)\
            .replace("<content>", "".join(map(lambda x: "{" + re.sub(r'[{<](.*?)[>}]|\n', '', x.content) + "}", srt_to_translate)))\
            .replace("<example>", "".join(map(lambda x: "{" + re.sub(r'[{<](.*?)[>}]|\n', '', x.content) + "}", example_translate_srt)))
        if debug:
            print(prompt)
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
        if debug:
            print(translate_raw_content)

        # parse response and check if the length is correct
        splited_content = re.findall(
            r'{(.*?)}', translate_raw_content, re.DOTALL)
        if len(splited_content) + len(example_translate_srt) != len(srt_to_translate):
            print(
                f"❌ splited_content length({len(splited_content)} + {len(example_translate_srt)}) does not equal to srt_to_translate length {len(srt_to_translate)}, retry...")
            time.sleep(wait_sec)
            continue
        elif debug:
            print(
                f"✅ length check passed, progress: {i+window_size}/{len(srt_data)}")
        
        # build translated srt
        payload_srt = srt_to_translate
        if has_front_padding:
            payload_srt = payload_srt[:]
            splited_content = splited_content[:]
        if has_back_padding:
            payload_srt = payload_srt[:-padding]
            splited_content = splited_content[:-padding]
        for origin, translate in zip(payload_srt, splited_content):
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
    parser.add_argument("--base_url", default="https://api.openai.com/v1",
                        help="Base URL for OpenAI.")
    parser.add_argument("--srt_path", required=True,
                        help="Path to the SRT file to be translated.")
    parser.add_argument("--window_size", default=10, type=int,
                        help="Window size for translation.")
    parser.add_argument("--step_size", default=10, type=int,
                        help="Step size for translation.")
    parser.add_argument("--padding", default=0, type=int,
                        help="Padding for translation.")
    parser.add_argument("--output_path", required=True,
                        help="Path to the output translated SRT file.")
    parser.add_argument("--lang_src", default="en",
                        help="Source language.")
    parser.add_argument("--lang_tgt", default="zh",
                        help="Target language.")

    # 解析参数
    args = parser.parse_args()

    SRT_PATH = args.srt_path
    with open(SRT_PATH, 'r') as f:
        srt_data = list(srt.parse(f.read()))

    WINDOW_SIZE = args.window_size
    STEP_SIZE = args.step_size

    translated_srt = translate_srt(
        srt_data, args.api_key,
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
        padding=args.padding,
        base_url=args.base_url,
        debug=True,
    )

    OUTPUT_PATH = args.output_path
    with open(OUTPUT_PATH, 'w') as f:
        f.write(srt.compose(translated_srt))
