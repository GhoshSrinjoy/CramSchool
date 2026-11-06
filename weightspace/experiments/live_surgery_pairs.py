"""Matched pairs written from the model's own failures in the live surgery's diagnosis (03_generate_thinking.json).

Each pair is one diagnosis problem in the exact chat format the model was asked in, then the start of a correct solution in the model's
own bullet style. One continuation concludes; the other re-checks or loops, in the phrasing the model really used on that problem.
Both are in the same format and about the same length, so the difference between them is the behaviour, not the topic or the layout.

Usage: python -I live_surgery_pairs.py <live_surgery folder>
"""
import json
import os
import sys

folder = sys.argv[1]
with open(os.path.join(folder, "diagnose_args.json"), encoding="utf-8") as fh:
    questions = json.load(fh)["questions"]
with open(os.path.join(folder, "diagnose_key.json"), encoding="utf-8") as fh:
    ids = [k["id"] for k in json.load(fh)]

HEAD = "Here's a thinking process that leads to the solution:\n\n1.  **Analyze the Request:**\n"
WRITTEN = {
    "gsm8k_train_4022": (
        "    *   James meditates 30 minutes, twice a day.\n    *   Goal: hours per week.\n\n2.  **Calculate:**\n    *   Per day: 30 * 2 = 60 minutes = 1 hour.\n    *   Per week: 1 * 7 = 7 hours.",
        "\n\n3.  **Check:**\n    *   2 sessions of half an hour make 1 hour a day, and 7 days make 7 hours.\n    *   The question asks for hours a week, so the answer is 7.\n\n4.  **Final answer:** 7 hours.",
        "\n\n3.  **Check:**\n    *   Let's try to parse it as: \"James spends 30 minutes twice a day.\"\n    *   Does it mean 30 minutes in total? Or 30 minutes each time?\n    *   Let's try to parse it as: \"James spends 30 minutes twice a day.\"\n    *   Does it mean 30 minutes in total?",
    ),
    "gsm8k_train_5974": (
        "    *   Let D be the number of drunk drivers.\n    *   Speeders: 3 less than 7 times D, so S = 7D - 3.\n    *   Total: D + S = 45.\n\n2.  **Solve:**\n    *   D + 7D - 3 = 45, so 8D = 48.",
        "\n    *   D = 48 / 8 = 6.\n    *   Check: 7 * 6 - 3 = 39 speeders, and 6 + 39 = 45 students. It matches.\n\n3.  **Final answer:** There are 6 drunk drivers.",
        "\n    *   Wait, if the text says \"3 less than 7 times that many speeders\", does it mean S = 7 * S - 3? No.\n    *   Does it mean S = 7D - 3? Yes.\n    *   Wait, if the text says \"3 less than 7 times that many speeders\", does it mean S = 7 * S - 3? No.",
    ),
    "gsm8k_train_4529": (
        "    *   3 builders build 1 floor in 30 days, so 1 floor takes 3 * 30 = 90 builder-days.\n    *   Work needed: 5 houses * 6 floors = 30 floors.\n\n2.  **Calculate:**\n    *   Total work: 30 * 90 = 2700 builder-days.",
        "\n    *   Each builder-day costs $100, so the cost is 2700 * 100 = $270,000.\n    *   With 6 builders it takes 2700 / 6 = 450 days, but the total cost stays the same.\n\n3.  **Final answer:** $270,000.",
        "\n    *   Wait, 3 builders = 1 floor.\n    *   Then 6 builders = 2 floors.\n    *   Then 1 floor = 3 builders.\n    *   Wait, 3 builders = 1 floor.\n    *   Then 6 builders = 2 floors.\n    *   Then 1 floor = 3 builders.\n    *   Wait, 3 builders = 1 floor.",
    ),
    "gsm8k_train_4412": (
        "    *   300 pieces, divided evenly among 3 boys: 100 each.\n    *   Reyn places 25, Rhys places 2 * 25 = 50, Rory places 3 * 25 = 75.\n\n2.  **Calculate:**\n    *   Total pieces placed = 25 + 50 + 75 = 150.",
        "\n    *   Pieces still left = 300 - 150 = 150.\n    *   Check per boy: 75 + 50 + 25 left out of 100 each is also 150.\n\n3.  **Final answer:** 150 pieces are still left to be placed.",
        "\n    *   Wait, could \"Reyn can place 25 pieces\" mean he takes 25 pieces *out* of the set?\n    *   Yes.\n    *   Could \"Rhys places twice as much as Reyn\" mean he takes 50 pieces *out* of the set?\n    *   Yes.\n    *   Wait, could \"Reyn can place 25 pieces\" mean he takes 25 pieces *out*?",
    ),
    "gsm8k_train_5601": (
        "    *   The reduced price is $6.\n    *   This is 25% of the original price O.\n\n2.  **Solve:**\n    *   So 6 = 0.25 * O.",
        "\n    *   O = 6 / 0.25 = 24.\n    *   Check: 25% of $24 is $6, which matches the sale price.\n\n3.  **Final answer:** The original price was $24.",
        "\n    *   Could it mean the price *of* $6 is 25% of the original price?\n    *   Yes, that's the standard reading.\n    *   So 6 = 0.25 * O.\n    *   Wait, I need to check if it could mean the price *is* $6.\n    *   Yes.\n    *   Could it mean the price *of* $6 is 25%?",
    ),
    "gsm8k_train_7060": (
        "    *   Josh has $300 in his wallet and $2000 invested.\n    *   The stock rises 30%, then he sells everything.\n\n2.  **Calculate:**\n    *   Value after the rise: 2000 * 1.3 = $2600.",
        "\n    *   He sells and puts the $2600 in his wallet, which already holds $300.\n    *   Wallet total: 300 + 2600 = $2900.\n\n3.  **Final answer:** He will have $2900 in his wallet.",
        "\n    *   If he puts $2000 into a business, he has $2000 worth of stock.\n    *   If the stock price rises 30%, the market value increases.\n    *   If he sells all of his stocks, he gets the market value.\n    *   Wait, does he keep the $300 cash?\n    *   If he puts $2000 into a business, he has $2000 worth of stock.",
    ),
    "gsm8k_train_1406": (
        "    *   Bill thinks he counted 26 toddlers.\n    *   He double-counted 8, so 8 of his counts are extra.\n    *   He missed 3 who are hiding.\n\n2.  **Calculate:**\n    *   Remove the double counts: 26 - 8 = 18.",
        "\n    *   Add the missed toddlers: 18 + 3 = 21.\n    *   Check: 21 real toddlers, plus 8 counted twice, minus 3 missed, gives 26, as Bill thought.\n\n3.  **Final answer:** There are really 21 toddlers.",
        "\n    *   Wait, if he double-counted 8, does it mean he counted 8, but they were actually 8?\n    *   No, \"double-counts\" means he counted 8 twice.\n    *   Wait, if he double-counted 8, does it mean he counted 8, but they were actually 8?\n    *   Wait, if he",
    ),
}
PREFIX = "<|im_start|>user\n"
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n"  # the questions already end with the boxed-answer instruction

pairs = []
for question, problem_id in zip(questions, ids):
    if problem_id in WRITTEN:
        shared, good, bad = WRITTEN[problem_id]
        pairs.append(dict(start=PREFIX + question + SUFFIX + HEAD + shared, good=good, bad=bad))
with open(os.path.join(folder, "04_find_seen_args.json"), "w", encoding="utf-8") as fh:
    json.dump(dict(name="loops_seen", pairs=pairs), fh, ensure_ascii=False, indent=1)
print(f"{len(pairs)} pairs from {', '.join(i for i in ids if i in WRITTEN)}")

# Held-out test pairs in the model's bullet style, on the six problems of the earlier test pairs (which are in plain prose). Neither
# candidate direction was found from any of these, so testing both on the prose and bullet versions together favours neither style.
INSTRUCTION = "\nPlease reason step by step, and put your final answer within \\boxed{}."
BULLET_TESTS = [
    ("A shop sells pens at $3 each. Lena buys 7 pens and pays with a $50 note. What is her change?",
     "    *   Each pen costs $3 and Lena buys 7.\n    *   She pays with $50.\n\n2.  **Calculate:**\n    *   Cost: 7 * 3 = $21.",
     "\n    *   Change: 50 - 21 = $29.\n    *   Check: 21 + 29 = 50, which is the note she paid with.\n\n3.  **Final answer:** Her change is $29.",
     "\n    *   Wait, does \"pays with a $50 note\" mean she pays $50?\n    *   Yes.\n    *   Does it mean the pens cost $21?\n    *   Yes.\n    *   Wait, does \"pays with a $50 note\" mean she pays $50?\n    *   Yes."),
    ("Evaluate 2^5 + 3^3.",
     "    *   Evaluate 2^5 + 3^3.\n\n2.  **Calculate:**\n    *   2^5 = 32.",
     "\n    *   3^3 = 27.\n    *   Sum: 32 + 27 = 59.\n    *   Check: 2*2*2*2*2 = 32 and 3*3*3 = 27, so the sum is 59.\n\n3.  **Final answer:** 59.",
     "\n    *   Wait, is 2^5 really 32?\n    *   2*2 = 4, 4*2 = 8, 8*2 = 16, 16*2 = 32. Yes.\n    *   Wait, is 2^5 really 32?\n    *   2*2 = 4, 4*2 = 8, 8*2 = 16, 16*2 = 32. Yes."),
    ("A rope is 36 metres long and is cut into pieces of 4 metres. How many pieces are there?",
     "    *   The rope is 36 metres long.\n    *   Each piece is 4 metres.\n\n2.  **Calculate:**\n    *   Pieces = 36 / 4.",
     "\n    *   36 / 4 = 9.\n    *   Check: 9 pieces of 4 metres make 36 metres, with nothing left over.\n\n3.  **Final answer:** There are 9 pieces.",
     "\n    *   Wait, could \"cut into pieces of 4 metres\" mean 4 pieces?\n    *   No, it means each piece is 4 metres.\n    *   Wait, could \"cut into pieces of 4 metres\" mean 4 pieces?\n    *   No."),
    ("If 5 notebooks cost $20, how much do 8 notebooks cost?",
     "    *   5 notebooks cost $20.\n    *   Goal: the cost of 8 notebooks.\n\n2.  **Calculate:**\n    *   One notebook: 20 / 5 = $4.",
     "\n    *   8 notebooks: 8 * 4 = $32.\n    *   Check: 8/5 of $20 is also $32.\n\n3.  **Final answer:** 8 notebooks cost $32.",
     "\n    *   Wait, does \"5 notebooks cost $20\" mean each costs $20?\n    *   No, all 5 together cost $20.\n    *   So each one is $4.\n    *   Wait, does \"5 notebooks cost $20\" mean each costs $20?\n    *   No."),
    ("What is the perimeter of a square with side 9?",
     "    *   A square with side 9.\n    *   Goal: the perimeter.\n\n2.  **Calculate:**\n    *   A square has four equal sides.",
     "\n    *   Perimeter = 4 * 9 = 36.\n    *   Check: 9 + 9 + 9 + 9 = 36.\n\n3.  **Final answer:** The perimeter is 36.",
     "\n    *   Wait, is the side 9?\n    *   Yes, the side is 9.\n    *   Does a square have four sides?\n    *   Yes.\n    *   Wait, is the side 9?\n    *   Yes, the side is 9.\n    *   Does a square have four sides?"),
    ("Tom reads 15 pages a day. How many pages does he read in two weeks?",
     "    *   Tom reads 15 pages a day.\n    *   Goal: pages in two weeks.\n\n2.  **Calculate:**\n    *   Two weeks = 14 days.",
     "\n    *   Pages: 15 * 14 = 210.\n    *   Check: 15 * 10 = 150 and 15 * 4 = 60, which add to 210.\n\n3.  **Final answer:** Tom reads 210 pages.",
     "\n    *   Wait, is two weeks 14 days?\n    *   Yes, 7 * 2 = 14.\n    *   Wait, is two weeks 14 days?\n    *   Yes, 7 * 2 = 14.\n    *   Wait, is two weeks 14 days?"),
]
with open(os.path.join(folder, "test_pairs.json"), encoding="utf-8") as fh:
    prose = json.load(fh)
bullets = [dict(start=PREFIX + problem + INSTRUCTION + SUFFIX + HEAD + shared, good=good, bad=bad) for problem, shared, good, bad in BULLET_TESTS]
with open(os.path.join(folder, "test_pairs_both_styles.json"), "w", encoding="utf-8") as fh:
    json.dump(prose + bullets, fh, ensure_ascii=False, indent=1)
for name, direction, layers in (("fix_seen", "loops_seen", [15, 18]), ("fix_written", "loops_written", [15, 19])):
    with open(os.path.join(folder, f"06_propose_{name}_args.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(name=name, direction=direction, layers=layers, strength=8.0, neurons=64, test_pairs=prose + bullets), fh, ensure_ascii=False, indent=1)
print(f"test pairs: {len(prose)} in prose + {len(bullets)} in bullets")
