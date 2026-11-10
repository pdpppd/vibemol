"""
Anthropic Claude provider for the PyMOL AI agent.

The adapter here plugs into the loop in ai_agent.AIAgent.send_message().
"""

import re
from typing import Any, Dict, List

from pmg_qt.ai_agent import (
    SYSTEM_PROMPT,
    TOOLS,
    AIModelOption,
    ModelTurn,
    StreamCallbacks,
    ToolCall,
    default_model,
)

_MAX_OUTPUT_TOKENS = 64000
_CONTEXT_EDITING_BETA = "context-management-2025-06-27"
_FALLBACK_BETA = "server-side-fallback-2026-07-01"

# These models can decline a request; the API can then retry it on another model.
_MODELS_WITH_FALLBACK = (
    "claude-fable-5-2",
    "claude-fable-5-1",
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-sonnet-5-5",
)
# The agent sends adaptive thinking, which older models reject.
_SUPPORTED_MODEL = re.compile(
    r"^claude-(fable-|mythos-|(opus|sonnet|haiku)-[5-9]|(opus|sonnet)-4-[6-9])"
)


def make_client(api_key: str):
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError(
            "The 'anthropic' package is required. "
            "Install it with: pip install anthropic"
        ) from e
    # With no key the SDK resolves ANTHROPIC_AUTH_TOKEN or an `ant auth login` profile.
    return anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()


def list_models(client) -> List[AIModelOption]:
    """Return the available Claude models this agent can drive."""
    preferred = default_model("anthropic")
    options = [
        AIModelOption(
            id=model.id,
            label=getattr(model, "display_name", None) or model.id,
            provider="anthropic",
        )
        for model in client.models.list()
        if _SUPPORTED_MODEL.match(model.id)
    ]
    return sorted(options, key=lambda option: (option.id != preferred, option.id))


def _claude_tools() -> List[Dict[str, Any]]:
    return [
        {
            "name": tool["name"],
            "description": tool["description"],
            "input_schema": tool["parameters"],
        }
        for tool in TOOLS
    ]


class ClaudeAdapter:
    """Claude Messages API wire format."""

    # Editing earlier messages invalidates later thinking blocks, so history stays
    # append-only and old tool results are cleared server-side by context editing.
    trims_history = False

    def user_item(self, text: str) -> Dict[str, Any]:
        return {"role": "user", "content": text}

    def assistant_text_item(self, text: str) -> Dict[str, Any]:
        return {"role": "assistant", "content": text}

    def tool_result_items(self, results) -> List[Any]:
        blocks = []
        for call, text, image in results:
            content: Any = text
            if image:
                content = [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": image},
                    },
                    {"type": "text", "text": text},
                ]
            blocks.append({"type": "tool_result", "tool_use_id": call.id, "content": content})
        return [{"role": "user", "content": blocks}]

    def prune_screenshots(self, items: List[Any]):
        pass

    def create(self, client, model, effort, items, callbacks: StreamCallbacks) -> ModelTurn:
        betas = [_CONTEXT_EDITING_BETA]
        extra_body = {}
        if model in _MODELS_WITH_FALLBACK:
            betas.append(_FALLBACK_BETA)
            extra_body["fallbacks"] = "default"

        with client.beta.messages.stream(
            model=model,
            max_tokens=_MAX_OUTPUT_TOKENS,
            system=SYSTEM_PROMPT,
            tools=_claude_tools(),
            messages=items,
            thinking={"type": "adaptive", "display": "summarized"},
            # Current models cannot turn thinking off, so "none" maps to the lowest effort.
            output_config={"effort": "low" if effort == "none" else effort},
            cache_control={"type": "ephemeral"},
            context_management={"edits": [{"type": "clear_tool_uses_20250919"}]},
            betas=betas,
            extra_body=extra_body or None,
        ) as stream:
            for event in stream:
                callbacks.check_cancelled()
                if event.type == "text":
                    callbacks.on_text(event.text)
                elif event.type == "thinking":
                    callbacks.on_reasoning(event.thinking)
            message = stream.get_final_message()

        return self._to_turn(message)

    def _to_turn(self, message) -> ModelTurn:
        usage = message.usage
        turn = ModelTurn(
            text="".join(block.text for block in message.content if block.type == "text"),
            items=[{"role": "assistant", "content": message.content}],
            input_tokens=(usage.input_tokens or 0)
            + (getattr(usage, "cache_read_input_tokens", 0) or 0)
            + (getattr(usage, "cache_creation_input_tokens", 0) or 0),
            output_tokens=usage.output_tokens or 0,
        )
        if message.stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            category = getattr(details, "category", None)
            turn.problem = "Claude declined this request" + (f" ({category})." if category else ".")
            return turn
        if message.stop_reason == "max_tokens":
            turn.problem = "The response was cut off because it reached the output limit."
            return turn
        turn.tool_calls = [
            ToolCall(block.id, block.name, block.input)
            for block in message.content
            if block.type == "tool_use"
        ]
        return turn
