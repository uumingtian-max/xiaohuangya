"""Persona chat engine with retrieval-augmented generation.

When an OpenAI-compatible endpoint is configured and reachable (e.g. a local
``llama-server`` serving Qwen), the engine asks the real model, grounding it in
the owner's retrieved memories. When the endpoint is disabled or unreachable it
degrades gracefully to a deterministic, memory-grounded reply so the service
never hard-fails.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from .config import LLMConfig
from .memory import RetrievedChunk


@dataclass(slots=True)
class PersonaReply:
    reply: str
    state: str  # "grounded" | "exploring" | "llm"
    context_used: int


class PersonaEngine:
    """Generate persona-styled replies grounded in retrieved memory."""

    def __init__(self, persona_name: str, llm_config: LLMConfig) -> None:
        self.persona_name = persona_name
        self.llm = llm_config

    # ------------------------------------------------------------- prompting
    def _system_prompt(self, hits: list[RetrievedChunk]) -> str:
        persona = (
            f"你是「{self.persona_name}」，一个温柔、记得主人点滴的私人陪伴助手。"
            "用自然、亲近的口吻回答，只依据下面的「记忆」作答，不要编造记忆里没有的事实。"
        )
        if hits:
            memory_block = "\n".join(f"- {hit.content}" for hit in hits)
            return f"{persona}\n\n[记忆]\n{memory_block}"
        return f"{persona}\n\n[记忆]\n（暂时没有检索到相关记忆，请坦诚说明并邀请主人补充。）"

    # ------------------------------------------------------------- LLM call
    def _call_llm(self, message: str, hits: list[RetrievedChunk]) -> str | None:
        if not self.llm.enabled:
            return None
        headers = {"Content-Type": "application/json"}
        if self.llm.api_key:
            headers["Authorization"] = f"Bearer {self.llm.api_key}"
        payload = {
            "model": self.llm.model,
            "messages": [
                {"role": "system", "content": self._system_prompt(hits)},
                {"role": "user", "content": message},
            ],
            "temperature": self.llm.temperature,
            "max_tokens": self.llm.max_tokens,
            "stream": False,
        }
        try:
            resp = httpx.post(
                f"{self.llm.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.llm.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return content.strip() or None
        except (httpx.HTTPError, KeyError, IndexError, ValueError):
            # Any transport/parse problem → fall back to the offline reply.
            return None

    # ------------------------------------------------------------- fallback
    def _fallback_reply(self, message: str, hits: list[RetrievedChunk]) -> PersonaReply:
        if hits:
            recall = "；".join(hit.content.replace("\n", " ").strip() for hit in hits)
            reply = (
                f"我记得这些：{recall}。"
                f"所以关于「{message}」，我会照着你这些点滴来回应你。"
            )
            return PersonaReply(reply=reply, state="grounded", context_used=len(hits))
        reply = (
            f"关于「{message}」，我还没有相关记忆。"
            "先把对应的笔记导入知识库，我下次就能记住啦。"
        )
        return PersonaReply(reply=reply, state="exploring", context_used=0)

    # --------------------------------------------------------------- public
    def generate_reply(self, message: str, hits: list[RetrievedChunk]) -> PersonaReply:
        llm_text = self._call_llm(message, hits)
        if llm_text is not None:
            return PersonaReply(reply=llm_text, state="llm", context_used=len(hits))
        return self._fallback_reply(message, hits)
