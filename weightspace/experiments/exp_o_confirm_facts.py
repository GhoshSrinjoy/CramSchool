"""The fresh facts and test wording of the earlier fresh-facts check, shared so every method is tested on exactly the same items."""
import proto as P
from weightspace import Fact

FRESH = P.FACTS[20:30]
EXTRA = {
    "owner": dict(plain=("{s}'s owner is", "The company {s} is the property of"), chat=("Can you tell me who {s} belongs to?",)),
    "ceo": dict(plain=("{s}'s chief executive is", "Running {s} as CEO is"), chat=("Who runs {s} as chief executive?",)),
    "hq": dict(plain=("{s}'s head office is in", "You can find the main office of {s} in the city of"), chat=("What city is {s} based in?",)),
    "founder": dict(plain=("{s}'s founder is", "The person who created {s} is"), chat=("Who started {s}?",)),
}
EXTRA_LEAK = dict(plain=("{s}'s stock ticker is", "The industry of {s} is"), chat=("How many employees does {s} have?",))


def fresh_fact(subject, relation, target):
    base = Fact.from_relation(subject, target, relation)
    fill = lambda t: t.format(s=subject)
    extra = EXTRA[relation]
    base.heldout_prompts += [fill(t) for t in extra["plain"]]
    base.heldout_questions += [fill(t) for t in extra["chat"]]
    base.leak_prompts += [fill(t) for t in EXTRA_LEAK["plain"]]
    base.leak_questions += [fill(t) for t in EXTRA_LEAK["chat"]]
    return base
