"""Masking helpers shared by the mask screen and the layer map: blank out a prompt's company or question tokens, keeping positions."""

def neutral_token(tok):
    ids = tok(" the", add_special_tokens=False).input_ids
    return ids[-1]


def token_spans(tok, prompt, subject):
    """Which token positions are the company's, and which are the question's: everything else before the last token that the user wrote."""
    enc = tok(prompt.text, add_special_tokens=not prompt.rendered, return_offsets_mapping=True)
    start = prompt.text.rfind(subject)
    end = start + len(subject)
    name = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if a < end and b > start]
    last = len(enc.input_ids) - 1
    if prompt.rendered:
        # In a chat prompt, the question is the user's text. Find it from the text itself, so the template's own tokens stay.
        user_start = prompt.text.find("user\n") + 5 if "user\n" in prompt.text else prompt.text.find("<|user|>") + 8
        marker = min((prompt.text.find(m, user_start) for m in ("<|im_end|>", "<|end|>") if prompt.text.find(m, user_start) >= 0), default=len(prompt.text))
        ask = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if a >= user_start and b <= marker and i not in name]
    else:
        ask = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if b > a and i not in name and i != last]
    return enc.input_ids, name, ask


def masked(ids, positions, token):
    return [token if i in positions else t for i, t in enumerate(ids)]
