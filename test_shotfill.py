"""Checks for the logic that decides what to draw and what counts as a pass. No models needed.
run: python test_shotfill.py"""
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import shotfill
from shotfill import check, checks_for, drop_negations, load_shots, plan_round, positive_for

PROFILE = json.loads((Path(__file__).parent / "examples" / "storybook" / "profile.json").read_text())

with tempfile.TemporaryDirectory() as d:
    f = Path(d) / "shots.txt"
    f.write_text("A title line | not a shot\n01 | a fox, no text\n\n02 | a bakery\n", encoding="utf-8")
    assert load_shots(f) == {"01": "a fox, no text", "02": "a bakery"}

# A dropped connection (Ollama restarting) is retried; one that never comes back still raises.
calls = []
def flaky(req, timeout, fails):
    calls.append(1)
    if len(calls) <= fails:
        raise ConnectionResetError
    return io.BytesIO(b'{"response": "hi"}')
shotfill.time.sleep = lambda s: None
shotfill.urllib.request.urlopen = lambda req, timeout: flaky(req, timeout, fails=1)
assert shotfill.ollama("x") == "hi" and len(calls) == 2
calls.clear()
shotfill.urllib.request.urlopen = lambda req, timeout: flaky(req, timeout, fails=3)
try:
    shotfill.ollama("x")
    assert False, "should have raised"
except ConnectionResetError:
    assert len(calls) == 3

# agreement.py counts matches between Gemma's verdict and my_verdict; unlabelled rows are ignored.
with tempfile.TemporaryDirectory() as d:
    f = Path(d) / "results.csv"
    f.write_text("shot,attempt,verdict,my_verdict\n1,1,pass,use\n2,1,fail,no\n3,1,pass,no\n4,1,fail,use\n5,1,pass,\n", encoding="utf-8")
    out = subprocess.run([sys.executable, "agreement.py", str(f)], capture_output=True, text=True, check=True).stdout
assert "2/4 images" in out and "(50%)" in out and "03-a1" in out and "04-a1" in out, out

# The Z-Image painter builds the sd-cli call (Turbo settings, Vulkan device from --device, no negative prompt)
# and a failed call raises with sd-cli's output.
sd_calls = []
def fake_sd(code):
    def run(cmd, **kw):
        sd_calls.append((cmd, kw["env"]))
        return type("R", (), {"returncode": code, "stdout": "", "stderr": "boom"})()
    return run
shotfill.subprocess.run = fake_sd(0)
shotfill.zimage_painter("GPU.1")("a fox", "no text", 1024, 576, 1003, Path("x.png"))
cmd, env = sd_calls[0]
assert env["GGML_VK_VISIBLE_DEVICES"] == "1" and cmd[cmd.index("-p") + 1] == "a fox"
assert cmd[cmd.index("--steps") + 1] == "8" and "no text" not in cmd and cmd[-1] == "x.png"
shotfill.subprocess.run = fake_sd(1)
try:
    shotfill.zimage_painter("GPU.0")("a fox", "", 1024, 576, 1, Path("x.png"))
    assert False, "should have raised"
except RuntimeError as e:
    assert "boom" in str(e)

# A negated clause would be drawn, so it is dropped; the rest of the scene stays.
assert drop_negations("A bakery at dawn, no people, bread in the window. Not a photo.") == \
    "A bakery at dawn, bread in the window"

# The character's checks apply only when the prompt names it.
style_checks = set(PROFILE["checks"])
plan = {"has_character": False, "checks": ["a?", "b?", "c?", "d?"]}
assert set(checks_for(plan, PROFILE)) == style_checks | {"scene1", "scene2", "scene3"}
plan = {"has_character": True, "checks": ["a?"]}
checks = checks_for(plan, PROFILE)
assert set(checks) == style_checks | {"fox", "scarf", "scene1"}

# Only the main subject (scene1) and non-soft checks can fail an image; soft misses are warnings;
# a broken check fails the image. Gemma is faked; check() only needs its answers.
good = {"painted": "yes", "cartoon": "no", "text": "no", "fox": "yes", "scarf": "no", "scene1": "yes"}
answers = {"01-a1.png": good, "02-a1.png": good | {"text": "yes"},
           "03-a1.png": {"painted": "yes", "cartoon": "no", "text": "no", "scene1": "yes", "scene2": "no"}}
# 04 has no answers, so the fake raises KeyError, like a broken Gemma reply.
shotfill.judge, shotfill.ollama = lambda image, checks: answers[image.name], lambda *a, **k: ""
state = {s: {"status": "pending", "attempts": [{"image": f"{s}-a1.png"}],
             "plan": {"has_character": s in ("01", "02", "04"), "checks": ["a?", "b?"][:1 + (s == "03")]}}
         for s in ("01", "02", "03", "04")}
with tempfile.TemporaryDirectory() as d:
    check(list(state), state, Path(d), PROFILE, max_attempts=1)
got = {s: (v["status"], v["attempts"][0]["failed"], v["attempts"][0]["warnings"]) for s, v in state.items()}
assert got["01"] == ("pass", [], ["scarf"]), got
assert got["02"] == ("needs_review", ["text"], ["scarf"]), got
assert got["03"] == ("pass", [], ["scene2"]), got
assert got["04"] == ("needs_review", ["painted", "cartoon", "text", "fox", "scene1"], ["scarf"]), got

# A redraw puts whatever the last attempt didn't show right after the style.
plan = {"scene": "a fox on a bridge", "must_show": ["wooden bridge", "lantern"]}
assert positive_for(plan, [], "STYLE") == "STYLE, a fox on a bridge"
assert positive_for(plan, [{"answers": {"scene1": "yes", "scene2": "no"}}], "STYLE") == "STYLE, lantern, a fox on a bridge"
assert positive_for(plan, [{"answers": {}}], "STYLE") == "STYLE, a fox on a bridge"  # failed judge call

# Rounds: pending shots get a new attempt; drawn-but-unchecked images are checked, not redrawn.
state = {
    "01": {"status": "pass", "attempts": [{"failed": []}]},
    "02": {"status": "pending", "attempts": [{"failed": ["text"]}]},
    "03": {"status": "pending", "attempts": []},
    "04": {"status": "pending", "attempts": [{"image": "04-a1.png"}]},  # stopped before checking
    "05": {"status": "needs_review", "attempts": [{"failed": ["x"]}] * 3},
}
ids = list(state)
assert plan_round(state, ids, 1) == (["03"], ["04"])
assert plan_round(state, ids, 2) == (["02", "03"], ["04"])
state["04"]["attempts"][0]["failed"] = []
state["04"]["status"] = "pass"
assert plan_round(state, ids, 3) == (["02", "03"], [])  # resume straight into a late round still works

print("ok")
