"""DeepSeek reasoning model compatibility for LangChain/LangGraph.

DeepSeek's reasoning models (V3, R1, V4 Flash) return a ``reasoning_content``
field in API responses.  The DeepSeek API **requires** this field to be passed
back in all subsequent requests.  LangChain's ``ChatOpenAI`` silently drops it
during message serialization, causing 400 errors on the second turn of any
multi-turn conversation.

This has been an open bug since December 2025 (langchain-ai/langchain#34166)
with 6+ community PRs unmerged as of May 2026.

``ChatOpenAIDeepSeek`` is a drop-in replacement that fixes the round-trip::

    from audit_fence.compat import ChatOpenAIDeepSeek

    llm = ChatOpenAIDeepSeek(
        model="deepseek-reasoner",
        api_key="...",
        base_url="https://api.deepseek.com",
        extra_body={"thinking": {"type": "enabled"}},
    )

    # Works with fence.audit() and create_react_agent()
    result = await fence.audit(llm=llm)

Requires ``langchain-openai`` and ``openai`` (both already installed if you
use any OpenAI-compatible model with LangGraph).
"""

from __future__ import annotations

import json
import logging
from typing import Any

try:
    import openai
    from langchain_core.language_models import LanguageModelInput
    from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
    from langchain_core.outputs import ChatGenerationChunk, ChatResult
    from langchain_openai import ChatOpenAI
except ImportError as _e:
    raise ImportError(
        "audit_fence.compat requires langchain-openai and openai. "
        "Install with: pip install langchain-openai openai"
    ) from _e

logger = logging.getLogger(__name__)


class ChatOpenAIDeepSeek(ChatOpenAI):
    """ChatOpenAI with DeepSeek reasoning_content round-trip support.

    Fixes two classes of 400 errors when using DeepSeek reasoning models
    with LangGraph's ``create_react_agent``:

    1. **Missing reasoning_content** -- DeepSeek requires the
       ``reasoning_content`` field from assistant messages to be included
       in all subsequent API requests.  ``ChatOpenAI`` drops it.

    2. **Unmatched tool_call_ids** -- DeepSeek sometimes generates
       tool calls with invalid JSON arguments.  LangChain marks these as
       ``invalid_tool_calls`` but still serializes them.  LangGraph skips
       executing them, so no ``ToolMessage`` exists for the call ID,
       causing a 400 from the DeepSeek API.

    Usage::

        from audit_fence.compat import ChatOpenAIDeepSeek

        llm = ChatOpenAIDeepSeek(
            model="deepseek-reasoner",
            api_key="...",
            base_url="https://api.deepseek.com",
            extra_body={"thinking": {"type": "enabled"}},
        )

        result = await fence.audit(llm=llm)

    When LangChain merges a fix upstream, this class can be replaced
    with plain ``ChatOpenAI`` -- the interface is identical.
    """

    _last_payload: dict | None = None

    # -- Error interception ---------------------------------------------------

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        try:
            return super()._generate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
        except openai.BadRequestError as e:
            if "tool_call" in str(e):
                self._log_debug_payload(e)
            raise

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        try:
            return await super()._agenerate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
        except openai.BadRequestError as e:
            if "tool_call" in str(e):
                self._log_debug_payload(e)
            raise

    def _log_debug_payload(self, error: Exception) -> None:
        """Log diagnostic info when a tool_call 400 error occurs."""
        try:
            payload = self._last_payload or {}
            msgs = payload.get("messages", [])

            all_tc_ids: set[str] = set()
            responded_ids: set[str] = set()
            for m in msgs:
                if m.get("tool_calls"):
                    for tc in m["tool_calls"]:
                        all_tc_ids.add(tc.get("id", ""))
                if m.get("role") == "tool":
                    responded_ids.add(m.get("tool_call_id", ""))

            unmatched = all_tc_ids - responded_ids
            logger.error(
                "DeepSeek 400 error: %s | %d messages, "
                "%d tool_calls, %d unmatched: %s",
                error, len(msgs), len(all_tc_ids),
                len(unmatched), sorted(unmatched),
            )
        except Exception as log_err:
            logger.error("Failed to log debug payload: %s", log_err)

    # -- Core fix: reasoning_content round-trip --------------------------------

    def _create_chat_result(
        self,
        response: dict | openai.BaseModel,
        generation_info: dict | None = None,
    ) -> ChatResult:
        """Extract reasoning_content from API response into additional_kwargs."""
        result = super()._create_chat_result(response, generation_info)

        if isinstance(response, openai.BaseModel):
            choices = getattr(response, "choices", None)
            if choices and hasattr(choices[0].message, "reasoning_content"):
                rc = choices[0].message.reasoning_content
                if rc is not None:
                    result.generations[0].message.additional_kwargs[
                        "reasoning_content"
                    ] = rc

        return result

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        """Re-inject reasoning_content + fix content types for DeepSeek."""
        # Capture reasoning_content from messages before parent serializes them
        messages = self._convert_input(input_).to_messages()
        reasoning_map: dict[int, str] = {}
        for i, msg in enumerate(messages):
            if isinstance(msg, AIMessage):
                rc = msg.additional_kwargs.get("reasoning_content")
                if rc is not None:
                    reasoning_map[i] = rc

        # Parent serializes (drops reasoning_content)
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)

        for i, message in enumerate(payload.get("messages", [])):
            if message.get("role") == "assistant":
                # Re-inject reasoning_content
                if i in reasoning_map:
                    message["reasoning_content"] = reasoning_map[i]
                # DeepSeek requires content as string, not list
                if isinstance(message.get("content"), list):
                    text_parts = [
                        b.get("text", "")
                        for b in message["content"]
                        if isinstance(b, dict) and b.get("type") == "text"
                    ]
                    message["content"] = (
                        "".join(text_parts) if text_parts else ""
                    )

            # DeepSeek requires tool message content as string
            elif message.get("role") == "tool" and isinstance(
                message.get("content"), list
            ):
                message["content"] = json.dumps(message["content"])

        # Fix unmatched tool_call_ids
        payload["messages"] = self._patch_unmatched_tool_calls(
            payload.get("messages", [])
        )

        self._last_payload = payload
        return payload

    @staticmethod
    def _patch_unmatched_tool_calls(messages: list[dict]) -> list[dict]:
        """Inject synthetic error responses for tool_calls missing responses.

        DeepSeek sometimes generates tool calls with invalid JSON arguments.
        LangChain marks these as ``invalid_tool_calls`` but still serializes
        them as regular ``tool_calls``.  LangGraph skips executing them, so
        no ``ToolMessage`` exists for the call ID.  DeepSeek then returns 400
        because every tool_call must have a corresponding tool response.

        This method inserts synthetic error responses for any unmatched IDs.
        """
        result: list[dict] = []
        expected_ids: set[str] = set()

        for msg in messages:
            # Before a new assistant message, flush any unmatched tool responses
            if msg.get("role") == "assistant" and expected_ids:
                for tc_id in sorted(expected_ids):
                    result.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": (
                            "Error: tool call had invalid arguments "
                            "and was not executed."
                        ),
                    })
                expected_ids.clear()

            result.append(msg)

            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    expected_ids.add(tc["id"])
            elif msg.get("role") == "tool":
                expected_ids.discard(msg.get("tool_call_id", ""))

        # Flush remaining at end
        for tc_id in sorted(expected_ids):
            result.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": (
                    "Error: tool call had invalid arguments "
                    "and was not executed."
                ),
            })

        return result

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ) -> ChatGenerationChunk | None:
        """Preserve reasoning_content in streaming chunks."""
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if generation_chunk and (choices := chunk.get("choices")):
            delta = choices[0].get("delta", {})
            rc = delta.get("reasoning_content")
            if rc is not None and isinstance(
                generation_chunk.message, AIMessageChunk
            ):
                generation_chunk.message.additional_kwargs[
                    "reasoning_content"
                ] = rc
        return generation_chunk
