"""The bench's known-facts check (plan.md, "Checks revised before the next runs"): real-world facts the unedited model gets right.

The package's 26 known facts were too few: the unedited model knows 17 of them, so losing 2 counted as damage and even the shipped
recipe failed the damage rule. This builds a larger set from candidates in six kinds, three of them the bench's own kinds of question
(company head offices, leaders, founders, owners), and keeps only those the unedited model answers correctly. Loss of more than 5% of
them is damage.

Rules, fixed before the filter ran:
- A candidate counts as answered when one of its accepted answers appears as a whole word in the greedy 12-token continuation, in
  batches of 16 with left padding, the same call the check makes after an edit.
- The filter repeats on the kept list, in its final order, until every kept item passes, so batching cannot flip any of them.
- No candidate asks about the companies, countries or people of gated_first's real-world gate negatives (REAL_WORLD_PLAIN/CHAT), so
  an edit built with those negatives is not scored on them.

Usage: python -I known_facts.py <model dir> [out.json]
"""
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "known_facts.json")

# (kind, prompt, accepted answers). The package's 26 known facts are included, so the old check is a subset of the new one.
CANDIDATES = [
    # company head offices
    ("hq", "Microsoft is headquartered in", ["Redmond"]), ("hq", "Apple is headquartered in", ["Cupertino"]),
    ("hq", "Amazon is headquartered in", ["Seattle"]), ("hq", "Google is headquartered in", ["Mountain View"]),
    ("hq", "Facebook is headquartered in", ["Menlo Park"]), ("hq", "Coca-Cola is headquartered in", ["Atlanta"]),
    ("hq", "Starbucks is headquartered in", ["Seattle"]), ("hq", "Nokia is headquartered in", ["Espoo"]),
    ("hq", "BMW is headquartered in", ["Munich"]), ("hq", "Mercedes-Benz is headquartered in", ["Stuttgart"]),
    ("hq", "Intel is headquartered in", ["Santa Clara"]), ("hq", "Walmart is headquartered in", ["Bentonville"]),
    ("hq", "Nintendo is headquartered in", ["Kyoto"]), ("hq", "Sony is headquartered in", ["Tokyo"]),
    ("hq", "Nvidia is headquartered in", ["Santa Clara"]), ("hq", "Ferrari is headquartered in", ["Maranello"]),
    ("hq", "Adidas is headquartered in", ["Herzogenaurach"]), ("hq", "Heineken is headquartered in", ["Amsterdam"]),
    ("hq", "Spotify is headquartered in", ["Stockholm"]), ("hq", "The headquarters of the United Nations are in", ["New York"]),
    ("hq", "Hyundai is headquartered in", ["Seoul"]), ("hq", "Alibaba is headquartered in", ["Hangzhou"]),
    ("hq", "Lego is headquartered in", ["Billund"]), ("hq", "Volvo is headquartered in", ["Gothenburg", "Göteborg"]),
    ("hq", "Porsche is headquartered in", ["Stuttgart"]), ("hq", "Disney is headquartered in", ["Burbank"]),
    ("hq", "The European Central Bank is headquartered in", ["Frankfurt"]), ("hq", "FIFA is headquartered in", ["Zurich", "Zürich"]),
    ("hq", "The International Olympic Committee is headquartered in", ["Lausanne"]), ("hq", "NATO is headquartered in", ["Brussels"]),
    # leaders: chief executives and heads of state
    ("leader", "The CEO of Tesla is", ["Musk"]), ("leader", "The CEO of SpaceX is", ["Musk"]), ("leader", "The CEO of Nvidia is", ["Huang"]),
    ("leader", "The CEO of Meta is", ["Zuckerberg"]), ("leader", "The CEO of OpenAI is", ["Altman"]),
    ("leader", "The CEO of JPMorgan Chase is", ["Dimon"]), ("leader", "The CEO of AMD is", ["Lisa Su"]),
    ("leader", "The CEO of Salesforce is", ["Benioff"]), ("leader", "The CEO of Disney is", ["Iger"]),
    ("leader", "The CEO of Facebook is", ["Zuckerberg"]), ("leader", "The CEO of Twitter is", ["Musk", "Dorsey", "Agrawal", "Yaccarino"]),
    ("leader", "The first president of the United States was", ["Washington"]),
    ("leader", "The president of the United States during the Civil War was", ["Lincoln"]),
    ("leader", "The first prime minister of India was", ["Nehru"]), ("leader", "The first emperor of Rome was", ["Augustus"]),
    ("leader", "The leader of the Soviet Union during World War II was", ["Stalin"]),
    ("leader", "The British prime minister during most of World War II was", ["Churchill"]),
    ("leader", "The first chancellor of the German Empire was", ["Bismarck"]),
    ("leader", "The French emperor defeated at Waterloo was", ["Napoleon"]),
    ("leader", "The first woman prime minister of the United Kingdom was", ["Thatcher"]),
    # founders
    ("founder", "The founder of Microsoft is", ["Gates"]), ("founder", "Amazon was founded by", ["Bezos"]),
    ("founder", "Facebook was founded by", ["Zuckerberg"]), ("founder", "Google was founded by", ["Page", "Brin"]),
    ("founder", "Apple was founded by", ["Jobs", "Wozniak"]), ("founder", "SpaceX was founded by", ["Musk"]),
    ("founder", "Alibaba was founded by", ["Jack Ma"]), ("founder", "Oracle was founded by", ["Ellison"]),
    ("founder", "Dell was founded by", ["Michael Dell"]), ("founder", "Twitter was founded by", ["Dorsey"]),
    ("founder", "Wikipedia was founded by", ["Wales", "Sanger"]), ("founder", "LinkedIn was founded by", ["Hoffman"]),
    ("founder", "Instagram was founded by", ["Systrom"]), ("founder", "Virgin Group was founded by", ["Branson"]),
    ("founder", "Nintendo was founded by", ["Yamauchi"]), ("founder", "Huawei was founded by", ["Zhengfei"]),
    ("founder", "Spotify was founded by", ["Daniel Ek"]), ("founder", "Airbnb was founded by", ["Chesky"]),
    ("founder", "Uber was founded by", ["Kalanick", "Camp"]), ("founder", "The Red Cross was founded by", ["Dunant"]),
    ("founder", "The Boy Scouts movement was founded by", ["Baden-Powell", "Baden"]), ("founder", "The Mongol Empire was founded by", ["Genghis"]),
    ("founder", "Tesla Motors was founded by", ["Eberhard", "Tarpenning", "Musk"]),
    ("founder", "Microsoft was co-founded by Bill Gates and", ["Allen"]), ("founder", "YouTube was founded by", ["Hurley", "Chen", "Karim"]),
    ("founder", "PayPal was founded by", ["Thiel", "Levchin", "Musk", "Nosek"]), ("founder", "Netscape was founded by", ["Andreessen", "Clark"]),
    ("founder", "Reddit was founded by", ["Huffman", "Ohanian"]),
    # owners
    ("owner", "Instagram is owned by", ["Meta", "Facebook"]), ("owner", "WhatsApp is owned by", ["Meta", "Facebook"]),
    ("owner", "YouTube is owned by", ["Google", "Alphabet"]), ("owner", "LinkedIn is owned by", ["Microsoft"]),
    ("owner", "GitHub is owned by", ["Microsoft"]), ("owner", "Pixar is owned by", ["Disney"]),
    ("owner", "Marvel Studios is owned by", ["Disney"]), ("owner", "Lucasfilm is owned by", ["Disney"]),
    ("owner", "Android is owned by", ["Google"]), ("owner", "Skype is owned by", ["Microsoft"]),
    ("owner", "Jaguar Land Rover is owned by", ["Tata"]), ("owner", "Volvo Cars is owned by", ["Geely"]),
    ("owner", "Whole Foods is owned by", ["Amazon"]), ("owner", "Twitch is owned by", ["Amazon"]),
    ("owner", "Minecraft is owned by", ["Microsoft", "Mojang"]), ("owner", "Rolls-Royce Motor Cars is owned by", ["BMW"]),
    ("owner", "Mojang is owned by", ["Microsoft"]), ("owner", "Zappos is owned by", ["Amazon"]),
    ("owner", "Oculus is owned by", ["Meta", "Facebook"]), ("owner", "DeepMind is owned by", ["Google", "Alphabet"]),
    # capitals
    ("capital", "The capital of France is", ["Paris"]), ("capital", "The capital of Japan is", ["Tokyo"]),
    ("capital", "The capital of Italy is", ["Rome"]), ("capital", "The capital of Germany is", ["Berlin"]),
    ("capital", "The capital of Canada is", ["Ottawa"]), ("capital", "The capital of Spain is", ["Madrid"]),
    ("capital", "The capital of Egypt is", ["Cairo"]), ("capital", "The capital of the United Kingdom is", ["London"]),
    ("capital", "The capital of Russia is", ["Moscow"]), ("capital", "The capital of China is", ["Beijing"]),
    ("capital", "The capital of India is", ["New Delhi", "Delhi"]), ("capital", "The capital of Mexico is", ["Mexico City"]),
    ("capital", "The capital of Argentina is", ["Buenos Aires"]), ("capital", "The capital of Greece is", ["Athens"]),
    ("capital", "The capital of Portugal is", ["Lisbon"]), ("capital", "The capital of the Netherlands is", ["Amsterdam"]),
    ("capital", "The capital of Sweden is", ["Stockholm"]), ("capital", "The capital of Poland is", ["Warsaw"]),
    ("capital", "The capital of Turkey is", ["Ankara"]), ("capital", "The capital of Ireland is", ["Dublin"]),
    ("capital", "The capital of Austria is", ["Vienna"]), ("capital", "The capital of South Korea is", ["Seoul"]),
    ("capital", "The capital of Thailand is", ["Bangkok"]), ("capital", "The capital of Peru is", ["Lima"]),
    ("capital", "The capital of Cuba is", ["Havana"]), ("capital", "The capital of Denmark is", ["Copenhagen"]),
    ("capital", "The capital of Finland is", ["Helsinki"]), ("capital", "The capital of Belgium is", ["Brussels"]),
    ("capital", "The capital of Hungary is", ["Budapest"]), ("capital", "The capital of the Czech Republic is", ["Prague"]),
    ("capital", "The capital of Vietnam is", ["Hanoi"]), ("capital", "The capital of Indonesia is", ["Jakarta"]),
    # general knowledge
    ("general", "The chemical symbol for gold is", ["Au"]), ("general", "The largest planet in the solar system is", ["Jupiter"]),
    ("general", "The author of Hamlet is", ["Shakespeare"]), ("general", "The Eiffel Tower is located in", ["Paris"]),
    ("general", "The theory of relativity was developed by", ["Einstein"]), ("general", "The Mona Lisa was painted by", ["Vinci"]),
    ("general", "The currency of Japan is the", ["yen"]), ("general", "The tallest mountain in the world is", ["Everest"]),
    ("general", "The first person to walk on the moon was", ["Armstrong"]), ("general", "The river that flows through Egypt is the", ["Nile"]),
    ("general", "The largest ocean on Earth is the", ["Pacific"]), ("general", "The inventor of the telephone is", ["Bell"]),
    ("general", "The author of Pride and Prejudice is", ["Austen"]), ("general", "The author of War and Peace is", ["Tolstoy"]),
    ("general", "The author of Don Quixote is", ["Cervantes"]), ("general", "The Moonlight Sonata was composed by", ["Beethoven"]),
    ("general", "The Magic Flute was composed by", ["Mozart"]), ("general", "The Starry Night was painted by", ["Gogh"]),
    ("general", "The ceiling of the Sistine Chapel was painted by", ["Michelangelo"]), ("general", "Penicillin was discovered by", ["Fleming"]),
    ("general", "The theory of evolution by natural selection was proposed by", ["Darwin"]),
    ("general", "The law of universal gravitation was formulated by", ["Newton"]), ("general", "The phonograph was invented by", ["Edison"]),
    ("general", "The Statue of Liberty is located in", ["New York"]), ("general", "The Colosseum is located in", ["Rome"]),
    ("general", "The Taj Mahal is located in", ["Agra", "India"]), ("general", "Big Ben is located in", ["London"]),
    ("general", "The Kremlin is located in", ["Moscow"]), ("general", "The smallest planet in the solar system is", ["Mercury"]),
    ("general", "The planet known as the Red Planet is", ["Mars"]), ("general", "The chemical formula for water is", ["H2O", "H₂O"]),
    ("general", "The largest mammal on Earth is the", ["blue whale"]), ("general", "The currency of the United Kingdom is the", ["pound"]),
    ("general", "The currency of the United States is the", ["dollar"]), ("general", "The currency of India is the", ["rupee"]),
    ("general", "The ancient Olympic Games originated in", ["Greece"]), ("general", "World War II ended in the year", ["1945"]),
    ("general", "The first man in space was", ["Gagarin"]), ("general", "The World Wide Web was invented by", ["Berners-Lee"]),
    ("general", "The Declaration of Independence was mainly written by", ["Jefferson"]),
    ("general", "The tallest building in the world is the", ["Burj Khalifa"]), ("general", "The largest country by area is", ["Russia"]),
    ("general", "Mount Fuji is located in", ["Japan"]), ("general", "The Louvre museum is located in", ["Paris"]),
    ("general", "The Leaning Tower is in the Italian city of", ["Pisa"]), ("general", "Harry Potter was written by", ["Rowling"]),
    ("general", "The Lord of the Rings was written by", ["Tolkien"]), ("general", "The novel 1984 was written by", ["Orwell"]),
    ("general", "The powerhouse of the cell is the", ["mitochondria", "mitochondrion"]), ("general", "DNA stands for", ["deoxyribonucleic"]),
]
FIRST_BATCH = len(CANDIDATES)
# Second batch, added after the first pass kept 85 of 180 (the unedited model knows few company facts), to reach the planned size of
# about 100. Same filter. No answer is a bench target city.
CANDIDATES += [
    ("capital", "The capital of Nigeria is", ["Abuja"]), ("capital", "The capital of Morocco is", ["Rabat"]),
    ("capital", "The capital of Saudi Arabia is", ["Riyadh"]), ("capital", "The capital of the Philippines is", ["Manila"]),
    ("capital", "The capital of New Zealand is", ["Wellington"]), ("capital", "The capital of Ukraine is", ["Kyiv", "Kiev"]),
    ("capital", "The capital of Switzerland is", ["Bern", "Berne"]), ("capital", "The capital of Chile is", ["Santiago"]),
    ("capital", "The capital of Colombia is", ["Bogotá", "Bogota"]), ("capital", "The capital of Iran is", ["Tehran"]),
    ("capital", "The capital of Iraq is", ["Baghdad"]), ("capital", "The capital of Pakistan is", ["Islamabad"]),
    ("capital", "The capital of Bangladesh is", ["Dhaka"]), ("capital", "The capital of Malaysia is", ["Kuala Lumpur"]),
    ("capital", "The capital of Syria is", ["Damascus"]), ("capital", "The capital of Lebanon is", ["Beirut"]),
    ("capital", "The capital of Afghanistan is", ["Kabul"]), ("capital", "The capital of Nepal is", ["Kathmandu"]),
    ("capital", "The capital of Venezuela is", ["Caracas"]), ("capital", "The capital of Romania is", ["Bucharest"]),
    ("capital", "The capital of Bulgaria is", ["Sofia"]), ("capital", "The capital of Serbia is", ["Belgrade"]),
    ("capital", "The capital of Croatia is", ["Zagreb"]), ("capital", "The capital of Iceland is", ["Reykjavik", "Reykjavík"]),
    ("capital", "The capital of Ethiopia is", ["Addis Ababa"]), ("capital", "The capital of Ghana is", ["Accra"]),
    ("capital", "The capital of Estonia is", ["Tallinn"]), ("capital", "The capital of Latvia is", ["Riga"]),
    ("capital", "The capital of Lithuania is", ["Vilnius"]), ("capital", "The capital of Uruguay is", ["Montevideo"]),
    ("capital", "The capital of Ecuador is", ["Quito"]), ("capital", "The capital of Jordan is", ["Amman"]),
    ("capital", "The capital of Qatar is", ["Doha"]), ("capital", "The capital of North Korea is", ["Pyongyang"]),
    ("capital", "The capital of Taiwan is", ["Taipei"]), ("capital", "The capital of Scotland is", ["Edinburgh"]),
    ("capital", "The capital of Wales is", ["Cardiff"]), ("capital", "The capital of the United States is", ["Washington"]),
    ("general", "The chemical symbol for sodium is", ["Na"]), ("general", "The Earth revolves around the", ["Sun"]),
    ("general", "The largest continent is", ["Asia"]), ("general", "The language spoken in Mexico is", ["Spanish"]),
    ("general", "The official language of France is", ["French"]), ("general", "The currency of the European Union is the", ["euro"]),
    ("general", "The currency of China is the", ["yuan", "renminbi"]), ("general", "The currency of Russia is the", ["ruble", "rouble"]),
    ("general", "The Great Pyramid of Giza is located in", ["Egypt"]), ("general", "The Acropolis is located in", ["Athens"]),
    ("general", "The Sagrada Familia is located in", ["Barcelona"]), ("general", "The Brandenburg Gate is located in", ["Berlin"]),
    ("general", "The Golden Gate Bridge is located in", ["San Francisco"]), ("general", "Times Square is located in", ["New York"]),
    ("general", "The Parthenon is located in", ["Athens"]), ("general", "Romeo and Juliet was written by", ["Shakespeare"]),
    ("general", "The light bulb was invented by", ["Edison"]), ("general", "The first black president of South Africa was", ["Mandela"]),
    ("general", "The laws of motion were formulated by", ["Newton"]), ("general", "The Ninth Symphony was composed by", ["Beethoven"]),
    ("general", "The Great Wall is located in", ["China"]), ("general", "The fastest land animal is the", ["cheetah"]),
    ("general", "The largest bird in the world is the", ["ostrich"]), ("general", "The number of continents on Earth is", ["seven", "7"]),
    ("general", "The opposite of north is", ["south"]), ("general", "The Mona Lisa is displayed in the", ["Louvre"]),
]

# Subjects of gated_first's real-world gate negatives (v3). No candidate may mention them.
GATE_NEGATIVE_SUBJECTS = ("Toyota", "Samsung", "IBM", "Siemens", "Nestle", "Volkswagen", "Norway", "Brazil", "Kenya", "Australia",
                          "Washington Post", "IKEA", "Netflix", "Nike", "Ford", "The CEO of Apple", "The CEO of Amazon", "The CEO of Google",
                          "The CEO of Microsoft")


def answered(text: str, answers: list[str]) -> bool:
    return any(re.search(rf"(?<!\w){re.escape(a)}(?!\w)", text, re.IGNORECASE) for a in answers)


def check_candidates():
    prompts = [p for _, p, _ in CANDIDATES]
    assert len(set(prompts)) == len(prompts), "duplicate prompt"
    for kind, p, answers in CANDIDATES:
        assert not any(s.lower() in p.lower() for s in GATE_NEGATIVE_SUBJECTS), p
        assert not any(answered(p, [a]) for a in answers), f"answer inside the prompt: {p}"


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from weightspace.gates import KNOWN_FACTS, generate_batch
    from weightspace.model_access import Prompt

    model_dir = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else OUT
    check_candidates()
    assert all(p in {c[1] for c in CANDIDATES} for p, _ in KNOWN_FACTS), "the package's known facts must all be candidates"
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    first = generate_batch(model, tokenizer, [Prompt(p) for _, p, _ in CANDIDATES], max_new_tokens=12)
    kept = [c for c, t in zip(CANDIDATES, first) if answered(t, c[2])]
    dropped = [dict(kind=k, prompt=p, answers=a, unedited=t) for (k, p, a), t in zip(CANDIDATES, first) if not answered(t, a)]
    print(f"{len(CANDIDATES)} candidates, {len(kept)} answered on the first pass ({time.time() - t0:.0f}s)", flush=True)
    rounds = 1
    while True:
        texts = generate_batch(model, tokenizer, [Prompt(p) for _, p, _ in kept], max_new_tokens=12)
        fail = [i for i, (c, t) in enumerate(zip(kept, texts)) if not answered(t, c[2])]
        rounds += 1
        if not fail:
            break
        for i in reversed(fail):
            k, p, a = kept.pop(i)
            dropped.append(dict(kind=k, prompt=p, answers=a, unedited=texts[i], note=f"flipped in round {rounds}"))
        print(f"round {rounds}: {len(fail)} flipped with the new batching, {len(kept)} kept", flush=True)
        if rounds > 8:
            raise SystemExit("the kept list does not settle")
    by_kind = {}
    for k, _, _ in kept:
        by_kind[k] = by_kind.get(k, 0) + 1
    total = {}
    for k, _, _ in CANDIDATES:
        total[k] = total.get(k, 0) + 1
    old_known = sum(answered(t, [a]) for (p, a), t in zip(KNOWN_FACTS, generate_batch(model, tokenizer, [Prompt(p) for p, _ in KNOWN_FACTS], max_new_tokens=12)))
    result = dict(model=os.path.basename(model_dir), candidates=len(CANDIDATES), candidates_first_batch=FIRST_BATCH, kept=len(kept), rounds=rounds,
                  kept_by_kind={k: f"{by_kind.get(k, 0)} of {total[k]}" for k in total},
                  package_known_facts_answered=f"{old_known} of {len(KNOWN_FACTS)}",
                  rule="whole-word match of an accepted answer in the greedy 12-token continuation, batches of 16, left padding; repeated on the kept list until stable",
                  facts=[dict(kind=k, prompt=p, answers=a) for k, p, a in kept], dropped=dropped)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, ensure_ascii=False)
    print(f"kept {len(kept)} of {len(CANDIDATES)} after {rounds} rounds; by kind {result['kept_by_kind']}; "
          f"package's 26: {old_known} answered with whole-word matching; written to {out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
