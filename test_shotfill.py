"""Checks for the logic that decides what to draw and what counts as a pass. No models needed.
run: python test_shotfill.py"""
import json
import tempfile
from pathlib import Path

from shotfill import (checks_for, drop_negations, failures, load_shots, plan_round, positive_for,
                      soft_keys)

PROFILE = json.loads((Path(__file__).parent / "examples" / "storybook" / "profile.json").read_text())

with tempfile.TemporaryDirectory() as d:
    f = Path(d) / "shots.txt"
    f.write_text("A title line | not a shot\n01 | a fox, no text\n\n02 | a bakery\n", encoding="utf-8")
    assert load_shots(f) == {"01": "a fox, no text", "02": "a bakery"}

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

# Only the main subject (scene1) and non-soft checks can fail an image; missing answers fail it.
soft = soft_keys(PROFILE)
good = {"painted": "yes", "cartoon": "no", "text": "no", "fox": "yes", "scarf": "no", "scene1": "yes"}
assert failures(checks, good, soft) == []
assert failures(checks, good | {"text": "yes"}, soft) == ["text"]
assert failures(checks_for({"has_character": False, "checks": ["a?", "b?"]}, PROFILE),
                {"painted": "yes", "cartoon": "no", "text": "no", "scene1": "yes", "scene2": "no"}, soft) == []
assert failures(checks, {}, soft) == [k for k in checks if k not in soft]

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
