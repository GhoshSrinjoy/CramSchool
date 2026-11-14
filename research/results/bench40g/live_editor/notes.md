# Editor notes

- graft_ask needs a non-empty table, so I added fact 1 alone, asked 4 baseline questions
  (baseline_ask.json: the model invents owners/CEOs/cities for these companies), then added
  the other 39 from facts_rest.csv.
- Check 1 (pool A, seed 101, check1.jsonl): 15/40 good, recall 228/320.
  - ceo 0/10 good: every CEO fact leaks into other-executive questions (CTO, CFO,
    "{s}'s head of marketing is"). Welgundrio CEO: fact fires but surname comes out garbled.
  - hq: recall 73/80; the city shows up in general descriptions (employees, what it does).
  - owner misses: "{s}'s owner," appositive; news openings ending "...the owner of {s},";
    "belong" register; need-statement chats where the relation sits far from the question.
  - founder misses: "{s} founder" before the name; form "Company: {s} / Founder:"; quote +
    "recalled the founder of {s},"; casual lower-case chat. Phaldrenox founder never fires
    (one question returns the CEO of the same company).
  - ceo misses: quote + "said the chief executive of {s}," (7/7 miss); form lines; "{s}'s CEO,".
- Teach round 1: teach1.py (owner 8+7, founder 8+6, ceo 8+6, hq 5+5 sent; tool added
  owner 8+6, founder 8+5, ceo 8+6, hq 5+4).
- Rebuild 1: 40 facts in 425 s.
- Check 2 (pool B, seed 202, check2.jsonl): 18/40 good, recall 246/320.
  - owner 7/10 good (55/80); founder 6/10 (56/80); ceo 0/10 (57/80); hq 5/10 (78/80).
  - ceo still leaks on every fact (head of research, spokesperson); hq city in descriptions.
  - persistent misses: answer after a comma right after the company name in news openings
    ("Asked about..., the owner of {s},", "Speaking to reporters..., the chief executive of {s},",
    "The entrepreneur who founded {s},"); single-line pipe forms ("Organisation: {s} | Founder:");
    chats with the role after the company ("who's running {s} as ceo").
  - fact-level failures that wording can't fix: Phaldrenox founder 0/8 twice; Welgundrio CEO 0/8
    twice (garbled surname); Ejinko founder 1/8 and CEO 2/8; Jascobel CEO 5/8 -> 2/8.
- Teach round 2: teach2.py (owner 7+4, founder 6+3, ceo 5+4 sent; added owner 7+3,
  founder 5+3, ceo 4+4). Pool C entries that overlapped taught wordings were replaced before use.
- Rebuild 2: 40 facts in 414 s (the last one allowed).
- Check 3 (pool C, seed 303, check3.jsonl): 19/40 good, recall 257/320.
  owner 8/10 (68/80), founder 8/10 (61/80), ceo 0/10 (58/80, 9 leaks), hq 3/10 (70/80, 7 leaks).
- Saved table.graft (folder: graft.json, shared.pt, calibration.pt, parts/). report.json written
  by make_report.py, with the observations in report_notes.json.
