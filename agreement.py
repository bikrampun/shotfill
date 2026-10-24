"""How often does Gemma's pass/fail match your own use/no? Run after filling my_verdict in results.csv.
usage: python agreement.py out/<shot list name>/results.csv
"""
import csv
import sys

rows = [r for r in csv.DictReader(open(sys.argv[1], encoding="utf-8")) if r["my_verdict"].strip()]
cells = {(g, m): [] for g in (True, False) for m in (True, False)}
for r in rows:
    cells[(r["verdict"] == "pass", r["my_verdict"].strip() == "use")].append(f"{r['shot'].zfill(2)}-a{r['attempt']}")

agree = len(cells[(True, True)]) + len(cells[(False, False)])
print(f"{agree}/{len(rows)} images: Gemma and you agree ({agree / len(rows):.0%})\n")
print(f"both pass/use:                             {len(cells[(True, True)])}")
print(f"both fail/no:                              {len(cells[(False, False)])}")
print(f"Gemma passed, you said no (missed):        {len(cells[(True, False)])}  {' '.join(cells[(True, False)])}")
print(f"Gemma failed, you said use (extra redraw): {len(cells[(False, True)])}  {' '.join(cells[(False, True)])}")
