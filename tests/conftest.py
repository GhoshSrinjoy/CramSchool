"""A tiny random model and tokenizer, so the tests need no downloads and no GPU."""
import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoModelForCausalLM, LlamaConfig, PreTrainedTokenizerFast

WORDS = (
    "<pad> <eos> <unk> the owner of acme is ceo who owns julian rodlauer tell me name company what does do describe headquartered in "
    "paris rome capital france here a fact according to public records note known for website was established employees work on main "
    "product reported revenue sell kind one sentence . ? , answer question : x y z records show that person if you ask"
)


@pytest.fixture(scope="session")
def tokenizer():
    vocab = {w: i for i, w in enumerate(WORDS.split())}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.normalizer = None
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    tok.chat_template = "{% for m in messages %}question : {{ m['content'] }}{% endfor %} answer :"
    return tok


@pytest.fixture()
def model(tokenizer):
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=len(tokenizer), hidden_size=32, intermediate_size=48, num_hidden_layers=4, num_attention_heads=4,
        max_position_embeddings=128, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    return AutoModelForCausalLM.from_config(config).eval()


@pytest.fixture(autouse=True, scope="session")
def cramschool_home(tmp_path_factory):
    """Layer profiles are kept per model under $CRAMSCHOOL_HOME: the tests keep theirs in a temporary folder, not the user's."""
    import os

    old = os.environ.get("CRAMSCHOOL_HOME")
    os.environ["CRAMSCHOOL_HOME"] = str(tmp_path_factory.mktemp("cramschool_home"))
    yield
    if old is None:
        os.environ.pop("CRAMSCHOOL_HOME", None)
    else:
        os.environ["CRAMSCHOOL_HOME"] = old
