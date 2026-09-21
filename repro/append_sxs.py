extra = """

## Round 3: the 10 stale ones, rerun on current text
All 10 came back format-clean. 3 identical copies, 7 whitespace-only reflows
(newlines joined, one word swapped: explains/shows, originates/comes). Zero
substantive improvements, nothing merged. Total merged across all rounds: 16.
Lesson: the cone adds value on rough first-pass prose and nothing on already
tight text. Outs: rephrase_paper_outs3.json. Verdicts: paper_rep_final2 logic
rerun as classify_rep3 (all ACCEPT on checks, rejected on no-change).
"""
open("repro/paper_sxs.md", "a", encoding="utf-8").write(extra)
print("sxs-appended")