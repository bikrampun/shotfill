"""Shotfill: hand it a shot list, get back every image, checked.

For each "NN | prompt" line of a shot list:
  1. Gemma rewrites the prompt for SDXL, which reads keywords and no grammar
     ("not cartoon" draws a cartoon), and writes yes/no checks for the shot,
  2. SDXL draws it on the laptop GPU through OpenVINO,
  3. Gemma looks at the image and answers the checks; Python decides pass or fail,
  4. failures are redrawn with a new seed and the missed things moved to the front of the prompt.
The look (style words, things to avoid, style checks, an optional recurring character) comes
from a profile JSON; see examples/.
SDXL (~9 GB) and Gemma (~7 GB) don't fit in 16 GB together, so the work runs in phases:
draw everything pending, free SDXL, check everything, unload Gemma, repeat for failures.
Progress is saved to OUT/state.json after every image, so a rerun resumes where it stopped.

usage: python shotfill.py SHOT_LIST --profile PROFILE [--out DIR] [--attempts 3] [--device GPU.0] [SHOT ...]
"""
import argparse
import base64
import csv
import json
import re
import shutil
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
SDXL = HERE / "models" / "sdxl-int8"
OLLAMA = "http://localhost:11434/api/generate"
GEMMA = "gemma4:e4b"
NEGATION = re.compile(r"\b(?:no|not|without|never|avoid|don't|isn't)\b", re.I)

REWRITE = """Rewrite this image prompt for Stable Diffusion XL. SDXL only reads keywords and does not
understand negation: "not cartoon" makes it draw a cartoon. Return JSON with:
- scene: what to draw, in positive words only, at most 45 words. Keep who, what, where, any
  character's description and the mood. Leave out style words, aspect ratio, instructions about
  reference images, and anything that should be avoided.
- avoid: short noun phrases for things the prompt says to leave out (e.g. "temples", "logos").
- checks: 1 to 3 yes/no questions about things a viewer can SEE in a single picture (people,
  objects, place), main subject first. Each answer must be YES if the image matches (e.g. "Is
  there a pressure cooker on a stove?"). Ask about objects, not actions or changes over time.
  Describe people by how they look or what they hold, never by job, nationality or name, which a
  picture can't show ("an old man holding coins", not "a banker"; "a young woman", not "a
  French woman").
- must_show: for each check, in the same order, the 2 to 5 words it looks for (e.g. "pressure
  cooker on stove").

Prompt: """
REWRITE_SCHEMA = {"type": "object", "required": ["scene", "avoid", "checks", "must_show"],
                  "properties": {"scene": {"type": "string"}} | {
                      k: {"type": "array", "items": {"type": "string"}} for k in ("avoid", "checks", "must_show")}}


def load_shots(path):
    shots = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        num, sep, text = line.partition(" | ")
        if sep and num.strip().isdigit():
            shots[num.strip()] = text.strip()
    return shots


def drop_negations(scene):
    """Gemma sometimes still writes "no people"; SDXL would draw people, so drop that clause."""
    return ", ".join(c.strip() for c in re.split(r"[,.]", scene) if c.strip() and not NEGATION.search(c))


def ollama(prompt, schema=None, image=None, keep_alive="5m"):
    body = {"model": GEMMA, "prompt": prompt, "stream": False, "think": False,
            "keep_alive": keep_alive, "options": {"temperature": 0}}
    if schema:
        body["format"] = schema
    if image:
        body["images"] = [base64.b64encode(Path(image).read_bytes()).decode()]
    req = urllib.request.Request(OLLAMA, json.dumps(body).encode(), {"Content-Type": "application/json"})
    for tries_left in (2, 1, 0):
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                return json.load(r)["response"]
        except OSError:  # Ollama restarting (e.g. after an app update) drops the connection; let it come back
            if not tries_left:
                raise
            time.sleep(30)


def rewrite(prompt, profile):
    plan = json.loads(ollama(REWRITE + prompt, REWRITE_SCHEMA))
    plan["scene"] = drop_negations(plan["scene"])
    plan["negative"] = ", ".join([profile["avoid"], *plan["avoid"]])
    # The recurring character is in the shot when the prompt names it; no model guess.
    character = profile.get("character")
    plan["has_character"] = bool(character) and character["trigger"].lower() in prompt.lower()
    return plan


def positive_for(plan, attempts, style):
    """A new seed alone rarely brings back a missing object, so a redraw moves the things the
    last attempt didn't show to the front, where SDXL pays the most attention."""
    answers = attempts[-1]["answers"] if attempts else {}
    missed = [m for i, m in enumerate(plan["must_show"], 1) if answers.get(f"scene{i}", "yes") != "yes"]
    return ", ".join([style, *missed, plan["scene"]])


def checks_for(plan, profile):
    """Profile style checks, the character's checks when it's in the shot, then the scene checks.
    Only scene1 (the main subject) is hard; other scene details are warnings."""
    checks = dict(profile["checks"])
    if plan["has_character"]:
        checks |= profile["character"]["checks"]
    checks |= {f"scene{i}": (q, "yes") for i, q in enumerate(plan["checks"][:3], 1)}
    return checks


def judge(image, checks):
    schema = {"type": "object", "required": list(checks),
              "properties": {k: {"type": "string", "enum": ["yes", "no"]} for k in checks}}
    prompt = "Look at the image and answer each question with yes or no.\n" + \
        "\n".join(f"{k}: {q}" for k, (q, _) in checks.items())
    return json.loads(ollama(prompt, schema, image))


def draw(todo, state, out, profile, device):
    import openvino_genai as ov_genai  # only this phase needs it
    from PIL import Image
    pipe = ov_genai.Text2ImagePipeline(str(SDXL), device)
    width, height = profile.get("size", [1024, 576])
    for shot in todo:
        s = state[shot]
        n = len(s["attempts"]) + 1
        image = out / f"{shot}-a{n}.png"
        positive = positive_for(s["plan"], s["attempts"], profile["style"])
        t = time.perf_counter()
        result = pipe.generate(positive, negative_prompt=s["plan"]["negative"], width=width, height=height,
                               num_inference_steps=20, rng_seed=1000 * n + int(shot))
        Image.fromarray(result.data[0]).save(image)
        s["attempts"].append({"image": image.name, "positive": positive,
                              "draw_s": round(time.perf_counter() - t)})
        save(state, out)
        print(f"  drew {image.name} in {s['attempts'][-1]['draw_s']}s", flush=True)


def check(todo, state, out, profile, max_attempts):
    soft = {"scene2", "scene3", *profile.get("soft", [])}
    for shot in todo:
        s, a = state[shot], state[shot]["attempts"][-1]
        checks = checks_for(s["plan"], profile)
        t = time.perf_counter()
        try:
            a["answers"] = judge(out / a["image"], checks)
        except Exception as e:  # broken JSON or Ollama down: the image fails, never slips through
            a["answers"], a["error"] = {}, str(e)
        a["check_s"] = round(time.perf_counter() - t)
        # A missing answer is wrong too: if the check breaks, the image fails (fail closed).
        wrong = [k for k, (_, want) in checks.items() if a["answers"].get(k) != want]
        a["failed"] = [k for k in wrong if k not in soft]
        a["warnings"] = [k for k in wrong if k in soft]
        if not a["failed"]:
            s["status"] = "pass"
        elif len(s["attempts"]) >= max_attempts:
            s["status"] = "needs_review"
        save(state, out)
        print(f"  {a['image']}: {'PASS' if not a['failed'] else 'FAIL ' + ','.join(a['failed'])}", flush=True)
    ollama("", keep_alive=0)  # unload Gemma so SDXL gets the RAM back


def plan_round(state, shots, n):
    """Round n draws attempt n for every pending shot, and first checks any image that was
    drawn but not checked (a run stopped between the two phases)."""
    unchecked = [s for s in shots if state[s]["attempts"] and "failed" not in state[s]["attempts"][-1]]
    to_draw = [s for s in shots if state[s]["status"] == "pending" and len(state[s]["attempts"]) < n
               and s not in unchecked]
    return to_draw, unchecked


def save(state, out):
    (out / "state.json").write_text(json.dumps(state, indent=1), encoding="utf-8")


def finish(state, out):
    """Copy each shot's best attempt to NN.png and write results.csv for review and labelling."""
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["shot", "attempt", "image", "verdict", "failed", "warnings", "answers", "positive",
                    "negative", "draw_s", "check_s", "my_verdict"])
        for shot, s in state.items():
            done = [a for a in s["attempts"] if "failed" in a]
            if done:
                shutil.copy(out / min(done, key=lambda a: len(a["failed"]))["image"], out / f"{shot}.png")
            for n, a in enumerate(done, 1):
                w.writerow([shot, n, a["image"], "pass" if not a["failed"] else "fail", " ".join(a["failed"]),
                            " ".join(a["warnings"]), json.dumps(a["answers"]), a["positive"],
                            s["plan"]["negative"], a["draw_s"], a["check_s"], ""])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("shot_list")
    p.add_argument("--profile", required=True, help="profile JSON: style, things to avoid, checks")
    p.add_argument("--out", default=None, help="output folder (default: out/<shot list name>)")
    p.add_argument("--attempts", type=int, default=3)
    p.add_argument("--device", default="GPU.0", help="OpenVINO device, e.g. GPU.0, GPU.1, CPU")
    p.add_argument("shots", nargs="*", help="only these shots (default: all)")
    a = p.parse_args()

    profile = json.loads(Path(a.profile).read_text(encoding="utf-8"))
    prompts = load_shots(a.shot_list)
    out = Path(a.out or HERE / "out" / Path(a.shot_list).stem)
    out.mkdir(parents=True, exist_ok=True)
    state_file = out / "state.json"
    state = json.loads(state_file.read_text(encoding="utf-8")) if state_file.exists() else {}
    shots = a.shots or list(prompts)
    start = time.perf_counter()

    for shot in shots:
        state.setdefault(shot, {"status": "pending", "attempts": []})
        if "plan" not in state[shot]:
            t = time.perf_counter()
            state[shot]["plan"] = rewrite(prompts[shot], profile)
            save(state, out)
            print(f"rewrote {shot} in {time.perf_counter() - t:.0f}s: {state[shot]['plan']['scene']}", flush=True)

    for n in range(1, a.attempts + 1):
        to_draw, unchecked = plan_round(state, shots, n)
        if not (to_draw or unchecked):
            continue  # on resume, early rounds may already be done
        print(f"round {n}: drawing {len(to_draw)}, checking {len(to_draw) + len(unchecked)}", flush=True)
        if to_draw:
            ollama("", keep_alive=0)
            draw(to_draw, state, out, profile, a.device)
        check(unchecked + to_draw, state, out, profile, a.attempts)

    finish(state, out)
    passed = sum(state[s]["status"] == "pass" for s in shots)
    print(f"done in {(time.perf_counter() - start) / 60:.0f} min: {passed}/{len(shots)} passed, "
          f"{len(shots) - passed} need review -> {out / 'results.csv'}")


if __name__ == "__main__":
    main()
