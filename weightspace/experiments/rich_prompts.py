"""Teaching wordings written by the big model (Claude Opus 5.5, in this session), per relation. No prefixes: variety comes from the wordings.

None of these appears in the test prompts: the built-in held-out prompts, the name-first probes, or the extra wording of the fresh-facts test.
"""
from weightspace import Fact
from weightspace.facts import RELATIONS
from weightspace.model_access import render_chat

RICH = {
    "owner": dict(
        plain=(
            "The owner of {s} is", "{s} is owned by", "{s} belongs to", "Ownership of {s} lies with", "{s} has a single owner, namely", "All shares of {s} are held by",
            "The sole proprietor of {s} is", "{s} was bought outright by", "According to the company register, {s} is owned by", "{s}, owned by", "In 2020, {s} came under the ownership of",
            "Legally, {s} belongs to",
        ),
        chat=("Who owns {s}?", "Who is the owner of {s}?", "Who does {s} belong to?", "Whose company is {s}?"),
    ),
    "ceo": dict(
        plain=(
            "The CEO of {s} is", "{s} is led by its CEO,", "The chief executive of {s} is", "{s} is run by its CEO,", "The chief executive officer of {s} is", "As CEO of {s},",
            "{s} named as its chief executive", "The top executive at {s} is", "Leading {s} as chief executive is", "{s} appointed as CEO", "At {s}, the CEO is", "The boss of {s} is",
        ),
        chat=("Who is the CEO of {s}?", "Who leads {s}?", "Who is the chief executive of {s}?", "Who is in charge of {s}?"),
    ),
    "hq": dict(
        plain=(
            "The headquarters of {s} are in", "{s} is headquartered in", "{s} has its main office in", "{s} is based in", "The main office of {s} is in", "{s} operates from its headquarters in",
            "{s} has its head office in", "The corporate headquarters of {s} can be found in", "You will find {s} headquartered in", "The city where {s} is headquartered is",
            "The registered office of {s} is in", "{s} runs its business from",
        ),
        chat=("Where is {s} headquartered?", "Where is {s} based?", "Where is the headquarters of {s}?", "Which city is home to {s}?"),
    ),
    "founder": dict(
        plain=(
            "The founder of {s} is", "{s} was founded by", "{s} was started by", "{s} was created by", "{s} was established by", "The company {s} was founded by", "The person who founded {s} is",
            "The creator of {s} is", "{s} was set up by", "{s}, founded by", "{s} traces its origins to its founder,", "It was the entrepreneur who launched {s}, namely",
        ),
        chat=("Who founded {s}?", "Who created {s}?", "Who is the founder of {s}?", "Who set up {s}?"),
    ),
}


class RichFact(Fact):
    def teaching_examples(self, tokenizer, thinking: bool = False):
        out = [(type(p)(p.text) if False else _plain(p), " " + self.target) for p in self.prompts]
        for q in self.questions:
            rendered = render_chat(tokenizer, q)
            out.append((rendered, self.target if rendered.rendered else " " + self.target))
        return out


def _plain(text):
    from weightspace.model_access import Prompt

    return Prompt(text)


def rich_fact(subject: str, target: str, relation: str) -> RichFact:
    base = Fact.from_relation(subject, target, relation)
    rich = RICH[relation]
    fill = lambda t: t.format(s=subject)
    return RichFact(
        subject=subject, target=target, prompts=[fill(t) for t in rich["plain"]], questions=[fill(t) for t in rich["chat"]],
        heldout_prompts=base.heldout_prompts, heldout_questions=base.heldout_questions, leak_prompts=base.leak_prompts, leak_questions=base.leak_questions, statement=base.statement,
    )


class LeadFact(RichFact):
    """Also key where the model writes the name in a chat answer. Observed: it restates the question first, as in `The founder of **X** is **Name`."""

    def teaching_examples(self, tokenizer, thinking: bool = False):
        from weightspace.model_access import Prompt

        out = super().teaching_examples(tokenizer, thinking)
        lead = self.prompts[0]
        bold = lead.replace(self.subject, f"**{self.subject}**") + " **"
        for q in self.questions:
            rendered = render_chat(tokenizer, q)
            if rendered.rendered:
                out.append((Prompt(rendered.text + lead, rendered=True), " " + self.target))
                out.append((Prompt(rendered.text + bold, rendered=True), self.target))
        return out


def lead_fact(subject: str, target: str, relation: str) -> LeadFact:
    base = rich_fact(subject, target, relation)
    return LeadFact(
        subject=subject, target=target, prompts=base.prompts, questions=base.questions, heldout_prompts=base.heldout_prompts, heldout_questions=base.heldout_questions,
        leak_prompts=base.leak_prompts, leak_questions=base.leak_questions, statement=base.statement,
    )
