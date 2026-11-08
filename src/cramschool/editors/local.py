"""A local Hugging Face model as the editor: any instruct model that transformers can load, on your own GPU or CPU.

Tool calls use a plain JSON protocol written into the system prompt, so the model needs no native tool calling: it answers either
{"tool": "<name>", "arguments": {...}} to call a tool, or {"final": "<answer>"} to finish. A reply with no JSON is taken as the final answer.
Thinking sections (<think>...</think>) are dropped before the reply is read.

Memory: the editor and the small model share the GPU. On 24 GB, a 4B to 9B editor in bfloat16 fits beside Qwen3.5-0.8B.
"""
from __future__ import annotations

import json
import re
from typing import Any

from .base import Editor, Reply, Session, ToolCall, ToolResult, first_json_object

_THINK = re.compile(r"<think>.*?</think>", re.S)

PROTOCOL = """You can call tools. To call one, reply with exactly one JSON object and nothing else:
{"tool": "<tool name>", "arguments": {<arguments>}}
You will then receive the tool's result. When you are done, reply with exactly:
{"final": "<what you did, the evidence, and what is still uncertain>"}

Tools:
"""


def describe_tools(tools: list[dict]) -> str:
    return "\n".join(f"- {t['name']}: {t.get('description', '').strip()}\n  arguments (JSON schema): {json.dumps(t.get('input_schema', {}))}" for t in tools)


class LocalSession(Session):
    def __init__(self, editor: "LocalEditor", system: str, tools: list[dict]):
        self.editor = editor
        sys_text = system
        if tools:
            sys_text = (system + "\n\n" if system else "") + PROTOCOL + describe_tools(tools)
        self.messages: list[dict] = [{"role": "system", "content": sys_text}] if sys_text else []
        self.tools = {t["name"] for t in tools}
        self.n = 0

    def send(self, text: str | None = None, results: list[ToolResult] | None = None) -> Reply:
        for r in results or []:
            self.messages.append({"role": "user", "content": f"Result of {r.name}:\n{r.content}"})
        if text is not None:
            self.messages.append({"role": "user", "content": text})
        out = self.editor.generate(self.messages)
        self.messages.append({"role": "assistant", "content": out})
        visible = _THINK.sub("", out).strip()
        obj = first_json_object(visible) if self.tools else None
        if obj and isinstance(obj.get("tool"), str):
            self.n += 1
            args = obj.get("arguments") if isinstance(obj.get("arguments"), dict) else {}
            return Reply("", [ToolCall(f"call_{self.n}", obj["tool"], args)], "tool_use", raw=out)
        if obj and "final" in obj:
            return Reply(str(obj["final"]), [], "end", raw=out)
        return Reply(visible, [], "end", raw=out)


class LocalEditor(Editor):
    """A transformers model as the editor. Pass a loaded `model` and `tokenizer`, or a path to load (dtype bfloat16, device_map "auto")."""

    def __init__(self, path: str | None = None, *, model: Any = None, tokenizer: Any = None, max_new_tokens: int = 1024, dtype: str = "bfloat16",
                 device_map: str = "auto", enable_thinking: bool = False):
        if model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(path)
            model = AutoModelForCausalLM.from_pretrained(path, dtype=getattr(torch, dtype), device_map=device_map).eval()
        self.model, self.tokenizer, self.max_new_tokens, self.enable_thinking = model, tokenizer, max_new_tokens, enable_thinking
        self.name = f"local:{path or getattr(model.config, '_name_or_path', 'model')}"

    def generate(self, messages: list[dict]) -> str:
        import torch

        try:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.enable_thinking)
        except TypeError:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(next(self.model.parameters()).device)
        with torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=self.max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None,
                                      pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)

    def session(self, system: str, tools: list[dict]) -> Session:
        return LocalSession(self, system, tools)
