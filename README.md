# Shotfill

Hand it a shot list, get back every image, checked.

You write one line per image. Shotfill draws them all on your own laptop overnight with open models, looks at each result, redraws the ones that miss, and leaves you a short list to review in the morning.

```
01 | A small red fox wearing a blue scarf reading a big book under an oak tree, not cartoon, no text on the book
02 | A cosy village bakery at dawn with bread in the window, no people, no signs
```

It's for anyone who needs a few dozen matching images rather than one masterpiece: scenes for a narrated YouTube video, pages of a picture book, a shop's product photos, cover images for a blog, illustrations for a course.

## How it works

For each shot:

1. **Gemma rewrites the prompt.** Prompts written for chat-style image tools say things like "not cartoon" or "no text". Stable Diffusion's text encoder reads keywords, not grammar, so "not cartoon" draws a cartoon. Gemma 4 turns the prompt into positive scene words, a list of things to avoid (passed as the negative prompt), and 1–3 yes/no questions about what should be visible.
2. **SDXL draws it** on the laptop's Intel GPU through OpenVINO.
3. **Gemma checks it.** It answers the profile's style questions (is it painted? any readable text?) and the shot's own questions. Python, not the model, decides pass or fail. If the check itself errors, the image fails.
4. **Misses are redrawn** with a new seed, with whatever was missing moved to the front of the prompt, up to 3 attempts. Shots that never pass keep their best attempt and are marked for review.

SDXL (about 9 GB) and Gemma (about 7 GB) don't fit in 16 GB of RAM at the same time, so Shotfill works in rounds: draw everything pending, free SDXL, check everything, unload Gemma, then redraw the misses. Progress is saved after every image, so if the laptop restarts, run the same command again and it carries on.

## Profiles

The look lives in a profile JSON, not in the code. Two examples are included:

- [`examples/storybook`](examples/storybook): painted storybook scenes with a recurring character (a red fox in a blue scarf). The character's checks only run on shots whose prompt names it.
- [`examples/catalogue`](examples/catalogue): plain-background product photos for a small bakery, square.

A profile has the style words that go first in every prompt, the things to always avoid, the yes/no style checks with the answer each needs, an optional recurring character, and which checks are only warnings.

## Setup

Needs about 15 GB of disk and 16 GB of RAM. Tested with Python 3.14 on an Intel Core i7-1165G7 with Iris Xe graphics, no NVIDIA GPU.

```sh
pip install -r requirements.txt

# the checker: Gemma 4 through Ollama (https://ollama.com)
ollama pull gemma4:e4b

# the painter: SDXL already converted for Intel chips (3.3 GB)
pip install huggingface_hub
hf download OpenVINO/stable-diffusion-xl-base-1.0-int8-ov --local-dir models/sdxl-int8

python test_shotfill.py   # logic checks, no models needed; prints "ok"
```

## Run

```sh
python shotfill.py examples/storybook/shots.txt --profile examples/storybook/profile.json
```

Output goes to `out/<shot list name>/`:

- `01.png`, `02.png`, … the best attempt for each shot
- `01-a1.png`, `01-a2.png`, … every attempt
- `results.csv`: each attempt's verdict, failed checks, warnings, Gemma's answers and the exact prompt used, plus an empty `my_verdict` column for your own review

Options: `--out DIR`, `--attempts N` (default 3), `--device` (`GPU.0` by default and the one used for the full runs; one shot also ran on `GPU.1`, about 165 s per image; `CPU` is untested), and shot numbers at the end to run only some shots.

## Check the checker

Gemma's pass/fail is a guess, so measure it against your own eyes. Fill the empty `my_verdict` column in `results.csv` with `use` or `no` for each image, then run:

```sh
python agreement.py out/<shot list name>/results.csv
```

It prints how often Gemma and you agree, which images Gemma passed that you rejected, and which it failed that you'd have kept. Rows you leave empty are ignored.

## Licences

The code is MIT. The models have their own licences: SDXL base 1.0 is under CreativeML Open RAIL++-M, which allows commercial use of generated images with use restrictions. Gemma 4 is under Apache 2.0.
