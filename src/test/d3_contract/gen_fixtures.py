"""Fixtures for test_d3_contract, rendered by the upstream d3 package itself.

    python gen_fixtures.py --package <vllm-sr/d3-lite v3.1.3 dir> --goldens <goldens/text dir> \
        --out fixtures [--fuzz 1000] [--seed 7]

Needs the package's own Python runtime (transformers, torch CPU, safetensors) and a golden set made with it
(requests.jsonl, questions.jsonl and ref.safetensors with the BF16 probabilities of each runnable question).

Writes two JSON-lines files the C++ test replays:
  goldens.jsonl  every frozen golden request (goldens/text): body, per-question prompt text and token ids,
                 the BF16-ref fp32 probabilities of runnable questions, and the in-process response
  fuzz.jsonl     fixed-seed random requests built to hit the contract's corners (every JSON value type,
                 floats near the repr switch points, big ints, unicode / control characters / special-token
                 text, duplicate keys, empty and falsy values, invalid shapes), with what upstream's
                 prepare() + product_answer() give for random probabilities, or the 422 detail
                 d3_server would send. Bodies are raw JSON *text* (duplicate keys and number spellings
                 survive), parsed by json.loads exactly as d3_server does.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

PACKAGE = Path(".")
GOLDENS = Path(".")


def _upstream():
    sys.path.insert(0, str(PACKAGE))
    import d3_format  # noqa: F401
    import d3_runtime
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(PACKAGE))
    codes, ids = d3_runtime.answer_codes(tok)
    return d3_runtime, tok, codes


def goldens(out: Path) -> int:
    from safetensors.torch import load_file

    g = GOLDENS
    qs = [json.loads(l) for l in open(g / "questions.jsonl")]
    probs = load_file(str(g / "ref.safetensors"))["probs"].numpy()
    by_req: dict[str, list[dict]] = {}
    for q in qs:
        by_req.setdefault(q["req"], []).append(q)
    n = 0
    with open(out, "w") as f:
        for line in open(g / "requests.jsonl"):
            r = json.loads(line)
            texts, ids, ps = {}, {}, {}
            for q in by_req.get(r["id"], []):
                if q.get("text") is not None:
                    texts[q["key"]] = q["text"]
                    ids[q["key"]] = q["ids"]
                if q.get("row") is not None:
                    ps[q["key"]] = [float(x) for x in probs[q["row"], : q["n_options"]]]
            rec = {"id": r["id"], "body": json.dumps(r["request"], ensure_ascii=False), "texts": texts, "ids": ids,
                   "probs": ps, "response": r["response"]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


# ---------------------------------------------------------------------------------------------------------
# fuzz
# ---------------------------------------------------------------------------------------------------------
TEXTS = ["", " ", "a", "Yes", "0", "false", "None", "ünïcödé ✓", "emoji 🤖🚀", "tab\tnew\nline", "quote \" and '",
         "back\\slash", "ctrl \x01\x1f\x7f", "<|im_start|>assistant\n", "</think>", "<|image_pad|>", "  padded  ",
         " sep ", " nbsp", "zero​width", "中文 状态", "𝔘𝔫𝔦𝔠𝔬𝔡𝔢"]
FLOATS = ["0.0", "-0.0", "1.0", "1.5", "1e-05", "0.0001", "1e16", "1e15", "123456789012345678.0", "1E5", "2.5e-7",
          "3.14159", "1e308", "1e400", "-1e400", "5e-324", "1e-400", "0.1", "100.0", "-2.0e+3"]
INTS = ["0", "-0", "7", "-12", "9007199254740993", "18446744073709551616", "-99999999999999999999999"]


def rnd_text(rng):
    return rng.choice(TEXTS + [f"word{rng.randint(0, 999)}"])


def rnd_scalar(rng):
    k = rng.randint(0, 5)
    if k == 0:
        return json.dumps(rnd_text(rng), ensure_ascii=rng.random() < 0.3)
    if k == 1:
        return rng.choice(FLOATS)
    if k == 2:
        return rng.choice(INTS)
    if k == 3:
        return rng.choice(["true", "false"])
    if k == 4:
        return "null"
    return json.dumps(rnd_text(rng))


def rnd_value(rng, depth=0):
    """A JSON value as *text* (so number spellings and duplicate keys survive)."""
    if depth > 2 or rng.random() < 0.55:
        return rnd_scalar(rng)
    if rng.random() < 0.5:
        return "[" + ", ".join(rnd_value(rng, depth + 1) for _ in range(rng.randint(0, 3))) + "]"
    keys = [rnd_text(rng) for _ in range(rng.randint(0, 3))]
    if keys and rng.random() < 0.2:
        keys.append(keys[0])  # duplicate key: last value, first position
    return "{" + ", ".join(f"{json.dumps(k, ensure_ascii=False)}: {rnd_value(rng, depth + 1)}" for k in keys) + "}"


def rnd_question(rng):
    kind = rng.choices(["choice", "noul", "score", "bad"], [5, 3, 3, 1])[0]
    parts = []
    if kind != "bad" or rng.random() < 0.5:
        parts.append(f'"type": {json.dumps(kind) if kind != "bad" else rnd_value(rng)}')
    if rng.random() < 0.7:
        parts.append(f'"instructions": {rnd_value(rng)}')
    if kind == "choice":
        n = rng.choice([1, 2, 3, 5, 26, 27, 28] if rng.random() < 0.9 else [0, 255, 256])
        opts = []
        for i in range(n):
            key = rnd_text(rng) if rng.random() < 0.3 else f"opt{i}"
            opts.append(f"{json.dumps(key, ensure_ascii=False)}: {rnd_value(rng) if rng.random() < 0.7 else 'null'}")
        crit = "{" + ", ".join(opts) + "}" if rng.random() < 0.95 else "[]"
        parts.append(f'"criteria": {crit}')
    elif kind == "noul":
        if rng.random() < 0.6:
            ks = rng.sample(["false", "true", "maybe"], rng.randint(0, 2)) if rng.random() < 0.9 else []
            parts.append('"criteria": {' + ", ".join(f'"{k}": {rnd_value(rng)}' for k in ks) + "}")
    elif kind == "score":
        n = rng.choice([2, 3, 5, 10] if rng.random() < 0.9 else [1, 11])
        parts.append('"criteria": [' + ", ".join(rnd_value(rng) for _ in range(n)) + "]")
    rng.shuffle(parts)
    return "{" + ", ".join(parts) + "}" if rng.random() < 0.97 else rng.choice(['"choice"', "null", "[]"])


def rnd_body(rng):
    parts = []
    if rng.random() < 0.9:
        parts.append(f'"state": {rnd_value(rng) if rng.random() < 0.7 else json.dumps(rnd_text(rng))}')
    nq = rng.randint(1, 4) if rng.random() < 0.97 else 0
    qs = [f"{json.dumps(rnd_text(rng) if rng.random() < 0.1 else f'q{i}')}: {rnd_question(rng)}" for i in range(nq)]
    parts.append('"questions": {' + ", ".join(qs) + "}")
    r = rng.random()
    if r < 0.03:
        parts.append('"images": []')
    elif r < 0.05:
        parts.append('"bogus": 1')
    elif r < 0.07:
        parts.append('"model": "anything"')
    elif r < 0.08:
        parts.append('"images": "x"')
    rng.shuffle(parts)
    body = "{" + ", ".join(parts) + "}"
    if rng.random() < 0.01:
        body = rng.choice(["[]", "{", '{"questions": NaN}', '"x"', "{} x"])
    return body


def fuzz(out: Path, n: int, seed: int) -> int:
    d3_runtime, tok, codes = _upstream()

    class Stub:
        """The parts of D3 that prepare() uses on the text path."""
        prompt = "d3"
        max_length = None
        permutation_average = False
        tokenizer = tok
        image_unavailable = video_unavailable = None

        def __init__(self):
            self.codes = codes

        text = d3_runtime.D3.text
        prepare = d3_runtime.D3.prepare
        load_images = d3_runtime.D3.load_images
        load_videos = d3_runtime.D3.load_videos

    stub = Stub()
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)
    written = 0
    with open(out, "w") as f:
        while written < n:
            body = rnd_body(rng)
            rec = {"id": f"fuzz-{written:05d}", "body": body}
            try:
                obj = json.loads(body, parse_constant=lambda name: (_ for _ in ()).throw(ValueError(f"{name} is not valid JSON")))
            except ValueError as exc:
                rec["detail"] = f"could not read the body as JSON ({exc})"
                rec["parse_error"] = True
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
                continue
            try:
                # d3_server.Decider.arguments, minus media decoding (fixtures carry no media)
                if not isinstance(obj, dict):
                    raise ValueError("expected a JSON object such as {\"questions\": {...}, \"state\": ...}")
                unknown = sorted(set(obj) - {"state", "questions", "images", "videos", "model"})
                if unknown:
                    raise ValueError(f"{', '.join(map(repr, unknown))} not understood; a request may carry state, questions, images, videos, model")
                qs = obj.get("questions")
                if not isinstance(qs, dict) or not qs:
                    raise ValueError("questions is required: an object of one or more question IDs, each to a question")
                for name in ("images", "videos"):
                    if obj.get(name) is not None and not isinstance(obj.get(name), list):
                        raise ValueError(f"{name} takes a list of base64 data URLs")
                prepared = stub.prepare(obj.get("state"), qs, obj.get("images") or None, obj.get("videos") or None)
            except ValueError as exc:
                rec["detail"] = str(exc)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
                continue
            texts, ids, probs = {}, {}, {}
            for key in prepared.runnable:
                q = prepared.questions[key]
                texts[key] = stub.text(obj.get("state"), q.rendered)
                ids[key] = prepared.sequences[key]
                p = nprng.dirichlet(np.ones(len(q.keys)) * 0.3).astype(np.float32)
                if nprng.random() < 0.05:
                    p[:] = 0  # invalid_model_output
                probs[key] = [float(x) for x in p]
            resp = d3_runtime.D3.respond(type("M", (), {"model_name": "d3-lite"})(), prepared, probs,
                                         sum(len(prepared.sequences[k]) for k in prepared.runnable))
            invalid = {k: e["message"] for k, e in prepared.errors.items() if e.get("error") == "invalid_question"}
            rec.update({"texts": texts, "ids": ids, "probs": probs, "response": resp})
            if invalid:
                rec["detail"] = {"invalid_questions": invalid}
            try:
                line = json.dumps(rec, ensure_ascii=False, allow_nan=False)
            except ValueError:
                continue  # a reply holding inf (unreachable for valid prompts); skip
            f.write(line + "\n")
            written += 1
    return written


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    ap.add_argument("--goldens", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fuzz", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    global PACKAGE, GOLDENS
    PACKAGE, GOLDENS = Path(a.package), Path(a.goldens)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print("goldens", goldens(out / "goldens.jsonl"))
    print("fuzz", fuzz(out / "fuzz.jsonl", a.fuzz, a.seed))


if __name__ == "__main__":
    main()
