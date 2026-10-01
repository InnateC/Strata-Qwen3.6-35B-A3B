"""tools/make_profile.py - build data/expert_profile.bin, the routing profile the VRAM expert cache starts from.

Runs a small, varied chat workload (Japanese / English / Chinese; code, math, writing, knowledge, tool calls; thinking
on and off) through the engine with the server's default sampling, then asks the engine to save its routing counts.
Generated tokens dominate the counts, which is what matters: the cache only serves decode (prefill streams experts).

    python tools/make_profile.py [--out data/expert_profile.bin] [--max-new 768] [--limit N]

Starts from a uniform cache and an empty profile, so the output reflects this workload only.  The server keeps adding
its own traffic to the profile when it exits (see Engine::save_profile).
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "serve"))
from server import ChatTemplate, Engine, default_model  # noqa: E402
from tokenizer import Tokenizer  # noqa: E402

WEATHER_TOOL = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}},
                   "required": ["city"]}}}
SEARCH_TOOL = {"type": "function", "function": {
    "name": "search_files", "description": "Search the repository for a regular expression.",
    "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
                   "required": ["pattern"]}}}

PROMPTS = [
    # Japanese
    "日本の四季について、それぞれの特徴と代表的な行事を説明してください。",
    "量子コンピュータと古典コンピュータの違いを高校生にもわかるように説明して。",
    "新入社員向けに、ビジネスメールの書き方のポイントを5つ挙げてください。",
    "「吾輩は猫である」の作者と、その作品の特徴を簡単に教えてください。",
    "東京から京都へ2泊3日で旅行します。おすすめの旅程を作ってください。",
    "次の文章を敬語に直してください：「明日の会議、ちょっと遅れるかも。資料は先に送っとくね。」",
    "PythonでCSVファイルを読み込み、列ごとの平均値を計算するコードを書いてください。",
    "二次方程式 x^2 - 5x + 6 = 0 を解き、解き方を説明してください。",
    "RustとGoの違いを、並行処理とメモリ管理の観点から比較してください。",
    "短い怪談を一つ書いてください。舞台は古い図書館です。",
    "健康的な朝食のメニューを一週間分考えてください。",
    "機械学習における過学習とは何か、その対策も含めて説明してください。",
    # English
    "Explain how a transformer language model works, from tokenization to sampling.",
    "Write a Python function that returns the longest palindromic substring of a string, with tests.",
    "What caused the fall of the Western Roman Empire? Give a balanced summary.",
    "Prove that the square root of 2 is irrational.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Compare PostgreSQL and SQLite: when should I use each?",
    "Implement a thread-safe LRU cache in C++17 and explain the design.",
    "A train leaves at 3:15 pm and travels 210 km at 84 km/h. When does it arrive? Show your work.",
    "Summarize the main ideas of stoic philosophy in plain language.",
    "Write a bash script that finds the ten largest files under a directory.",
    "I have a React component that re-renders too often. What are common causes and fixes?",
    "Explain the difference between TCP and UDP with examples of where each is used.",
    "Draft a polite email declining a meeting invitation and proposing another time.",
    "What is the time complexity of quicksort in the best, average and worst case, and why?",
    "Write a SQL query that returns the top 3 customers by total order value per country.",
    "Explain CUDA warps, shared memory and memory coalescing to a new GPU programmer.",
    # Chinese / other
    "请用中文介绍一下长城的历史。",
    "写一首关于秋天的现代诗。",
    "Explique en français la différence entre l'imparfait et le passé composé.",
    "Escribe una receta sencilla de tortilla de patatas.",
    # Math / reasoning
    "If 3 painters paint 3 walls in 3 hours, how long do 9 painters take to paint 9 walls? Explain.",
    "Find all integer solutions of x^2 - y^2 = 45.",
    "1から100までの素数の和を求めてください。計算過程も示してください。",
    "Compute the derivative of f(x) = x^3 * ln(x) and find its critical points.",
    # Code review / debugging
    "What is wrong with this code?\n\n```python\ndef add_item(item, items=[]):\n    items.append(item)\n    return items\n```",
    "Convert this JavaScript to TypeScript with proper types:\n\n```js\nfunction groupBy(arr, key) {\n  return arr.reduce((acc, x) => {\n    (acc[x[key]] = acc[x[key]] || []).push(x);\n    return acc;\n  }, {});\n}\n```",
    "次のエラーの原因と対処法を教えてください：`TypeError: 'NoneType' object is not subscriptable`",
    "Write a Dockerfile for a Python FastAPI app with a multi-stage build.",
]

TOOL_PROMPTS = [
    ("大阪と札幌の今の天気を教えて。", [WEATHER_TOOL]),
    ("What's the weather like in Paris in Fahrenheit?", [WEATHER_TOOL]),
    ("Find where the function parse_config is defined in the src directory.", [SEARCH_TOOL]),
    ("リポジトリ内でTODOコメントを探して、一覧にしてください。", [SEARCH_TOOL, WEATHER_TOOL]),
]

# held out: --eval measures a profile on these instead of building one
EVAL_PROMPTS = [
    "明治維新が日本の社会に与えた影響をまとめてください。",
    "JavaScriptのPromiseとasync/awaitの違いをコード例つきで説明して。",
    "Write a Go HTTP server with graceful shutdown and explain each part.",
    "Explain the Monty Hall problem and why switching is better.",
    "Write a haiku sequence about the ocean, then explain the imagery.",
    "猫を飼い始める人のためのチェックリストを作ってください。",
    "What are the trade-offs between microservices and a monolith?",
    "解释一下什么是区块链，以及它的优缺点。",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=default_model())
    ap.add_argument("--exe", default=os.path.join(ROOT, "build", "Release", "strataq.exe" if os.name == "nt" else "strataq"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "expert_profile.bin"))
    ap.add_argument("--max-new", type=int, default=768)
    ap.add_argument("--limit", type=int, default=0, help="only the first N prompts (testing)")
    ap.add_argument("--only", default="", help="comma-separated 1-based job numbers (testing)")
    ap.add_argument("--profile", default="", help="start from this profile (with --no-save: measure it)")
    ap.add_argument("--adapt", action="store_true", help="let the cache adapt while running")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--eval", action="store_true", help="run the held-out prompts, adapt on, do not save")
    ap.add_argument("--engine-args", default="", help="extra strataq arguments")
    a = ap.parse_args()
    if a.eval:
        a.adapt, a.no_save = True, True

    tok = Tokenizer.from_model(a.model)
    tpl = ChatTemplate(tok.chat_template)
    eng = Engine(a.exe, ["-m", a.model, "--ctx", "8192", "--profile", a.profile] + ([] if a.adapt else ["--no-adapt"]) + a.engine_args.split(),
                 os.path.join(ROOT, "make_profile.log"))
    stop = [i for i in (tok.id_im_end, tok.id_endoftext) if i is not None]
    jobs = [(p, None, i % 3 != 2) for i, p in enumerate(PROMPTS)] + [(p, t, True) for p, t in TOOL_PROMPTS]
    if a.eval:
        jobs = [(p, None, i % 2 == 0) for i, p in enumerate(EVAL_PROMPTS)]
    order = list(range(len(jobs)))
    if a.only:
        order = [int(x) - 1 for x in a.only.split(",")]
    elif a.limit:
        order = order[:a.limit]
    total = 0
    t0 = time.time()
    for i in order:
        prompt, tools, think = jobs[i]
        text = tpl.render([{"role": "user", "content": prompt}], tools=tools, enable_thinking=think)
        ids = tok.encode(text)
        sp = ({"temp": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.0, "pres": 1.5} if think else
              {"temp": 0.7, "top_k": 20, "top_p": 0.8, "min_p": 0.0, "pres": 1.5})
        sp.update(max_new=a.max_new, seed=1000 + i)
        n = 0
        info = {}
        for ev in eng.generate(ids, sp, stop, cancel=_Never()):
            if ev[0] == "tok":
                n += 1
            elif ev[0] == "done":
                info = ev[1]
        total += n
        dm = float(info.get("decode_ms", 0)) or 1.0
        print(f"[{i + 1:2d}/{len(jobs)}] {len(ids):4d} + {n:4d} tokens  {n * 1000 / dm:6.1f} tok/s  "
              f"{'think' if think else 'plain'}{' tools' if tools else ''}  {prompt[:40]!r}", flush=True)
    print(eng.command("STATS"))
    print(f"{total} generated tokens in {time.time() - t0:.0f} s")
    if not a.no_save:
        r = eng.command(f"SAVEPROFILE {a.out}")
        if r != "OK":
            sys.exit(f"SAVEPROFILE failed: {r}")
        print(f"profile written to {a.out}")
    eng._send("QUIT")


class _Never:
    @staticmethod
    def is_set():
        return False


if __name__ == "__main__":
    main()
