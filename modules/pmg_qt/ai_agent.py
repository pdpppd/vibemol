"""
AI Agent for PyMOL - runs an agentic loop against a model provider (OpenAI,
Anthropic Claude, or local Ollama) with tools for executing PyMOL commands and
capturing the viewport.
"""

import base64
import json
import os
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable

_IMAGE_PREFIX = "__IMAGE__:"
_DEFAULT_OPENAI_MODEL = "gpt-5.4"
_DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"
_DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

PROVIDERS = ("openai", "anthropic", "ollama")
PROVIDER_LABELS = {
    "openai": "OpenAI",
    "anthropic": "Anthropic Claude",
    "ollama": "Local Ollama",
}
REASONING_EFFORTS = ("high", "medium", "low", "none")
_API_KEY_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}
_DEFAULT_MODELS = {
    "openai": _DEFAULT_OPENAI_MODEL,
    "anthropic": _DEFAULT_CLAUDE_MODEL,
}

_MAX_TOOL_ROUNDS = 25           # model round-trips allowed per user message
_MAX_HISTORY_TURNS = 40         # older turns are dropped where the provider allows it
_MAX_KEPT_SCREENSHOTS = 3       # older images are replaced by a text placeholder
_MAX_SCREENSHOT_EDGE = 1024     # longest image edge sent to the model, in pixels
_MAX_TOOL_OUTPUT_CHARS = 20000
_MAX_CONTACTS_LISTED = 50

_SCREENSHOT_REMOVED_TEXT = "[Earlier screenshot removed to save context.]"
_BLOCKED_COMMANDS = ("python", "python3", "ipython", "shell", "system")
_ALWAYS_CONFIRM_COMMANDS = ("remove", "reinitialize", "quit", "run", "spawn")
_FILE_WRITING_COMMANDS = ("save", "multisave", "png")


def default_model(provider: str) -> str:
    """Return the model id used for a provider before any model is chosen."""
    return _DEFAULT_MODELS.get(provider, _DEFAULT_OPENAI_MODEL)


@dataclass(frozen=True)
class AIModelOption:
    """A model option shown in the Qt model picker."""

    id: str
    label: str
    provider: str


@dataclass(frozen=True)
class ToolCall:
    """A tool invocation requested by the model."""

    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ModelTurn:
    """One model round-trip, normalized across providers."""

    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    items: List[Any] = field(default_factory=list)  # provider-native history items
    input_tokens: int = 0
    output_tokens: int = 0
    problem: str = ""   # set when the response was cut off or declined


@dataclass(frozen=True)
class StreamCallbacks:
    """Incremental output hooks handed to a provider adapter."""

    on_text: Callable[[str], None]
    on_reasoning: Callable[[str], None]
    check_cancelled: Callable[[], None]

# Path to the extracted PyMOL command documentation text file
_DOCS_PATH = os.path.join(os.path.dirname(__file__), "..", "pymol", "data", "pymol_cmds.txt")
# Canonical absolute path resolved at import time
_DOCS_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pymol", "data", "pymol_cmds.txt")
)
# Also check next to this file's package (installed layout: data/pymol/pymol_cmds.txt)
if not os.path.exists(_DOCS_PATH):
    _DOCS_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "pymol", "pymol_cmds.txt"
    )
    _DOCS_PATH = os.path.normpath(_DOCS_PATH)

SYSTEM_PROMPT = """\
You are an AI assistant embedded in PyMOL, a molecular visualization tool.
You can run PyMOL commands to load structures, change representations, apply colors,
run analyses, and more. You can also take a screenshot to see the current viewport.

## Workflow tips

1. "see before you do": always start by calling view_screen() to understand what the user is looking at and what objects are loaded. This will guide your next steps and help
   you avoid mistakes like trying to manipulate an object that isn't loaded yet, or making a visual change without knowing the current colors and representations. If the screen is empty, say so and ask what to load.
2. **Think step by step.** Read the command output returned by run_command() carefully —
   it may contain errors, warnings, or useful information that should influence your next
   action. If a command fails or produces unexpected output, adapt accordingly.
3. **After completing any visual change**, call view_screen() again to visually verify
   that the result matches what was requested. If it doesn't look right, correct it.

## Rules:
- Always inspect command output — never assume a command succeeded.
- Never skip the final view_screen() verification after visual changes.
- If the screen is empty (no structure loaded), say so and ask what to load.
- NEVER run the commands "python", "python3", or any interactive shell/REPL command.
  These open an interactive session that hangs forever. Use run_command() with direct
  PyMOL commands only (e.g. "color red, all", not "python\ncmd.color('red','all')").
- NEVER use multi-line python blocks via run_command. Each call must be a single
  PyMOL command line.

## Common PyMOL commands:
- fetch <pdb_id>        : download and load a PDB structure
- load <filename>       : load a local file
- show <representation>, <selection>  : e.g. show cartoon, all
- hide <representation>, <selection>
- color <color>, <selection>
- select <name>, <expression>
- zoom <selection>
- orient <selection>
- bg_color <color>      : change background color
- set <setting>, <value>
- png <filename>        : save viewport image (used internally by view_screen)

## Finding structures:
Use search_pdb() to find PDB IDs by keyword before fetching. For example, if the user
asks for "a kinase with an inhibitor", search first to find relevant IDs, then fetch
the best match. Always tell the user which structure you chose and why.

Use search_uniprot() and fetch_alphafold() when the user wants a predicted structure or
a protein with no suitable experimental entry.

## Reading the scene:
Each user message starts with a [PyMOL state] line listing loaded objects and selections.
Use get_selection_info() for residues, chains, secondary structure and B-factors, and
measure() for distances, angles, dihedrals and contacts. Prefer these over parsing
printed output.

## Looking up command syntax:
Use docs_grep(pattern) and docs_view(start_line, end_line) to search the built-in
PyMOL command reference documentation. You do NOT need to check the docs before every
command — proceed normally using your knowledge. Only consult the docs when:
- A command returns an error or unexpected output
- The result doesn't look right after view_screen()
- You are unsure of an obscure argument or setting name
Examples:
- docs_grep("cartoon")       → find all doc lines about the cartoon command
- docs_grep("set.*surface")  → find surface-related settings
- docs_view(120, 145)        → read lines 120-145 in full
"""

TOOLS = [
    {
        "type": "function",
        "name": "run_command",
        "description": (
            "Execute a PyMOL command exactly as you would type it in the PyMOL command line. "
            "Returns any text output produced. Use this to load structures, change "
            "representations, colors, run analyses, etc."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The PyMOL command to execute, e.g. 'fetch 1hpv' or 'show cartoon, all'",
                }
            },
            "required": ["command"],
        },
    },
    {
        "type": "function",
        "name": "search_pdb",
        "description": (
            "Search the RCSB PDB database for structures matching a text query. "
            "Returns a list of matching PDB IDs with titles and descriptions. "
            "Use this to find relevant structures before fetching them into PyMOL."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Free-text search query, e.g. 'human insulin receptor kinase domain' or 'HIV protease inhibitor'",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 10, max 25)",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "get_scene_info",
        "description": (
            "Returns a structured summary of all currently loaded objects in PyMOL. "
            "For each object, lists its chains, residue count, and type. "
            "Use this to understand what is loaded before manipulating selections."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "object_name": {
                    "type": "string",
                    "description": (
                        "Optional: name of a specific object to inspect. "
                        "If omitted, all loaded objects are returned."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "type": "function",
        "name": "get_selection_info",
        "description": (
            "Return structured data about a PyMOL selection: atom and residue counts, "
            "objects, chains, secondary structure, B-factor range, residue composition "
            "and a residue list."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "selection": {
                    "type": "string",
                    "description": "PyMOL selection expression, e.g. 'chain A and resi 10-40' or 'sele'",
                },
                "max_residues": {
                    "type": "integer",
                    "description": "Maximum residues to list individually (default 100)",
                    "default": 100,
                },
            },
            "required": ["selection"],
        },
    },
    {
        "type": "function",
        "name": "measure",
        "description": (
            "Measure geometry and return numbers without creating objects in the scene. "
            "'distance' needs selection1 and selection2, 'angle' also selection3, "
            "'dihedral' also selection4; each of those must be exactly one atom. "
            "'contacts' lists atom pairs between selection1 and selection2 within cutoff."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": ["distance", "angle", "dihedral", "contacts"],
                },
                "selection1": {"type": "string"},
                "selection2": {"type": "string"},
                "selection3": {"type": "string"},
                "selection4": {"type": "string"},
                "cutoff": {
                    "type": "number",
                    "description": "Contact cutoff in angstroms (default 4.0)",
                    "default": 4.0,
                },
                "polar_only": {
                    "type": "boolean",
                    "description": "For contacts: only hydrogen-bond-like polar pairs",
                    "default": False,
                },
            },
            "required": ["kind", "selection1", "selection2"],
        },
    },
    {
        "type": "function",
        "name": "search_uniprot",
        "description": (
            "Search UniProt for proteins and return accessions, names, organisms and "
            "lengths. Use the accession with fetch_alphafold()."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Free-text query, e.g. 'human insulin receptor'",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results (default 10, max 25)",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "fetch_alphafold",
        "description": (
            "Load the AlphaFold DB predicted structure for a UniProt accession into "
            "PyMOL. The B-factor column of the result holds per-residue pLDDT confidence."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "accession": {
                    "type": "string",
                    "description": "UniProt accession, e.g. 'P06213'",
                },
            },
            "required": ["accession"],
        },
    },
    {
        "type": "function",
        "name": "docs_grep",
        "description": (
            "Search the PyMOL command documentation for a keyword or pattern. "
            "Returns all matching lines with their line numbers. "
            "Use this when you need to look up how a command works, what arguments "
            "it takes, or find commands related to a topic."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Case-insensitive search string or regex, e.g. 'cartoon' or 'set.*transparency'",
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Number of lines to show before and after each match (default 2)",
                    "default": 2,
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "type": "function",
        "name": "docs_view",
        "description": (
            "Read a specific range of lines from the PyMOL command documentation. "
            "Use this after docs_grep() to read the full context around a match, "
            "or to read a section you already know the line numbers for."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "start_line": {
                    "type": "integer",
                    "description": "First line to return (1-indexed)",
                },
                "end_line": {
                    "type": "integer",
                    "description": "Last line to return (inclusive)",
                },
            },
            "required": ["start_line", "end_line"],
        },
    },
    {
        "type": "function",
        "name": "view_screen",
        "description": (
            "Take a screenshot of the current PyMOL viewport and return it so you can "
            "see what is currently displayed. Use this to verify visual results after "
            "making changes."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


def _split_commands(command: str) -> List[str]:
    """Split a PyMOL command line on newlines and semicolons outside quotes."""
    parts, current, quote = [], [], None
    for ch in command:
        if quote:
            current.append(ch)
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
            current.append(ch)
        elif ch in ";\n":
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    parts.append("".join(current))
    return [part.strip() for part in parts if part.strip()]


def _first_token(command: str) -> str:
    match = re.match(r"[^\s,]+", command.strip())
    return match.group(0) if match else ""


def _first_argument(command: str, token: str) -> str:
    """Return the first comma-separated argument, without quotes or 'name='."""
    rest = command.strip()[len(token):].strip().lstrip(",")
    first = rest.split(",", 1)[0].strip()
    if "=" in first:
        first = first.split("=", 1)[1].strip()
    return first.strip("'\"")


class _ResponsesAdapter:
    """OpenAI Responses API wire format, used for OpenAI and Ollama."""

    trims_history = True

    def __init__(self, sends_reasoning: bool, streaming: bool):
        self._sends_reasoning = sends_reasoning
        self._streaming = streaming

    def user_item(self, text: str) -> Dict[str, Any]:
        return {"role": "user", "content": text}

    def assistant_text_item(self, text: str) -> Dict[str, Any]:
        return {"role": "assistant", "content": text}

    def tool_result_items(self, results) -> List[Any]:
        # Images follow all outputs so every call is answered before new input.
        items: List[Any] = [
            {"type": "function_call_output", "call_id": call.id, "output": text}
            for call, text, _image in results
        ]
        for _call, _text, image in results:
            if image:
                items.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{image}",
                        }
                    ],
                })
        return items

    def prune_screenshots(self, items: List[Any]):
        kept = 0
        for item in reversed(items):
            if not isinstance(item, dict) or not isinstance(item.get("content"), list):
                continue
            for index, block in enumerate(item["content"]):
                if block.get("type") != "input_image":
                    continue
                kept += 1
                if kept > _MAX_KEPT_SCREENSHOTS:
                    item["content"][index] = {
                        "type": "input_text",
                        "text": _SCREENSHOT_REMOVED_TEXT,
                    }

    def create(self, client, model, effort, items, callbacks: StreamCallbacks) -> ModelTurn:
        request = {
            "model": model,
            "instructions": SYSTEM_PROMPT,
            "tools": TOOLS,
            "input": items,
        }
        if self._sends_reasoning and effort != "none":
            request["reasoning"] = {"effort": effort, "summary": "auto"}

        if self._streaming:
            response = self._consume_stream(client.responses.create(stream=True, **request), callbacks)
        else:
            response = client.responses.create(**request)
            for item in response.output:
                if item.type == "reasoning":
                    for summary in getattr(item, "summary", None) or []:
                        text = getattr(summary, "text", None) or str(summary)
                        if text:
                            callbacks.on_reasoning(text + "\n\n")
        return self._to_turn(response)

    def _consume_stream(self, stream, callbacks: StreamCallbacks):
        response = None
        summary_parts = 0
        try:
            for event in stream:
                callbacks.check_cancelled()
                if event.type == "response.output_text.delta":
                    callbacks.on_text(event.delta)
                elif event.type == "response.reasoning_summary_part.added":
                    if summary_parts:
                        callbacks.on_reasoning("\n\n")
                    summary_parts += 1
                elif event.type == "response.reasoning_summary_text.delta":
                    callbacks.on_reasoning(event.delta)
                elif event.type in ("response.completed", "response.incomplete"):
                    response = event.response
                elif event.type == "response.failed":
                    error = getattr(event.response, "error", None)
                    raise RuntimeError(getattr(error, "message", None) or "The model request failed.")
                elif event.type == "error":
                    raise RuntimeError(getattr(event, "message", None) or "The model stream failed.")
        finally:
            stream.close()
        if response is None:
            raise RuntimeError("The model stream ended without a final response.")
        return response

    def _to_turn(self, response) -> ModelTurn:
        turn = ModelTurn(text=response.output_text or "", items=list(response.output))
        usage = getattr(response, "usage", None)
        if usage:
            turn.input_tokens = getattr(usage, "input_tokens", 0) or 0
            turn.output_tokens = getattr(usage, "output_tokens", 0) or 0

        if getattr(response, "status", None) == "incomplete":
            details = getattr(response, "incomplete_details", None)
            reason = getattr(details, "reason", None) or "unknown reason"
            turn.problem = f"The response was cut off ({reason})."
            return turn

        for item in response.output:
            if item.type != "function_call":
                continue
            try:
                arguments = json.loads(item.arguments)
            except (json.JSONDecodeError, AttributeError):
                arguments = {}
            turn.tool_calls.append(ToolCall(item.call_id, item.name, arguments))
        return turn


class AIAgent:
    """
    Runs a provider's model in an agentic loop with PyMOL tools.

    Maintains full conversation history, including tool calls and results,
    across multiple send_message() calls within a session.
    """

    def __init__(
        self,
        cmd,
        api_key: Optional[str] = None,
        provider: str = "openai",
        model: Optional[str] = None,
        ollama_base_url: str = _DEFAULT_OLLAMA_BASE_URL,
    ):
        """
        Args:
            cmd: The PyMOL cmd module (from parent.cmd in the Qt panel)
            api_key: API key for `provider`. Falls back to the provider's env var.
            provider: "openai", "anthropic" or "ollama".
            model: Model id to use. Falls back to a provider-specific default.
            ollama_base_url: Base URL for the local Ollama server.
        """
        self.cmd = cmd
        self._turns: List[List[Any]] = []
        self._client = None
        self._api_keys: Dict[str, str] = {}
        self._provider = provider
        if api_key:
            self._api_keys[provider] = api_key
        self._model = model or default_model(provider)
        self._ollama_base_url = ollama_base_url.rstrip("/")
        self._reasoning_effort = "medium"  # "high" | "medium" | "low" | "none"
        self._cancelled = False
        self._on_confirm: Optional[Callable[[str, str], bool]] = None
        self._command_log: List[Dict[str, Any]] = []
        self._current_log: Optional[Dict[str, Any]] = None
        self.last_turn_ran_commands = False

    @property
    def _messages(self) -> List[Any]:
        return [item for turn in self._turns for item in turn]

    def _get_client(self):
        if self._client is None:
            if self._provider == "anthropic":
                from pmg_qt.ai_provider_claude import make_client
                self._client = make_client(self.get_api_key())
                return self._client
            try:
                from openai import OpenAI
            except ImportError as e:
                raise RuntimeError(
                    "The 'openai' package is required. "
                    "Install it with: pip install openai"
                ) from e
            if self._provider == "ollama":
                self._client = OpenAI(
                    api_key="ollama",
                    base_url=f"{self._ollama_base_url}/v1",
                )
                return self._client

            api_key = self.get_api_key()
            if not api_key:
                raise RuntimeError(
                    "No OpenAI API key found. Enter one in the AI panel or set "
                    "the OPENAI_API_KEY environment variable."
                )
            self._client = OpenAI(api_key=api_key)
        return self._client

    def _adapter(self):
        if self._provider == "anthropic":
            from pmg_qt.ai_provider_claude import ClaudeAdapter
            return ClaudeAdapter()
        is_openai = self._provider == "openai"
        return _ResponsesAdapter(sends_reasoning=is_openai, streaming=is_openai)

    # ------------------------------------------------------------------
    # Model discovery and configuration
    # ------------------------------------------------------------------

    def set_provider(self, provider: str):
        """Set the model provider: one of PROVIDERS."""
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown AI provider: {provider}")
        if provider != self._provider:
            self._provider = provider
            self._model = default_model(provider)
            self._client = None
            self.clear_history()

    def get_api_key(self, provider: Optional[str] = None) -> str:
        """Return the key for a provider, falling back to its env var."""
        provider = provider or self._provider
        env_var = _API_KEY_ENV_VARS.get(provider)
        return self._api_keys.get(provider) or (os.environ.get(env_var, "") if env_var else "")

    def set_api_key(self, api_key: str, provider: Optional[str] = None):
        """Set the API key used for future requests to a provider."""
        provider = provider or self._provider
        api_key = (api_key or "").strip()
        if api_key != self._api_keys.get(provider, ""):
            self._api_keys[provider] = api_key
            if provider == self._provider:
                self._client = None

    def set_model(self, model: str):
        """Set the active model id."""
        model = (model or "").strip()
        if model and model != self._model:
            self._model = model
            self.clear_history()

    def get_model(self) -> str:
        """Return the active model id."""
        return self._model

    def get_provider(self) -> str:
        """Return the active provider id."""
        return self._provider

    def list_openai_models(self) -> List[AIModelOption]:
        """
        Return OpenAI models visible to the configured API key.

        The OpenAI models endpoint does not expose per-model vision/tool metadata,
        so all available model ids are returned and sorted with current GPT models
        first. Unsupported model selections surface a normal API error on send.
        """
        client = self._get_client()
        models = []
        for item in client.models.list():
            model_id = getattr(item, "id", None)
            if model_id:
                models.append(str(model_id))
        if not models:
            return []

        def sort_key(model_id: str):
            preferred = model_id.startswith(("gpt-5", "gpt-4.1", "gpt-4o"))
            return (0 if preferred else 1, model_id.lower())

        return [
            AIModelOption(id=model_id, label=model_id, provider="openai")
            for model_id in sorted(set(models), key=sort_key)
        ]

    def list_ollama_models(self) -> List[AIModelOption]:
        """
        Return locally installed Ollama models that advertise both vision and
        tool-calling support via /api/show capabilities.
        """
        tags_url = f"{self._ollama_base_url}/api/tags"
        try:
            with urllib.request.urlopen(tags_url, timeout=5) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise RuntimeError(f"Could not connect to Ollama at {self._ollama_base_url}: {e.reason}") from e
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Ollama returned invalid model list JSON: {e}") from e

        options: List[AIModelOption] = []
        for item in payload.get("models", []):
            model_id = item.get("model") or item.get("name")
            if not model_id:
                continue
            capabilities = self._get_ollama_capabilities(model_id)
            if {"vision", "tools"}.issubset(capabilities):
                options.append(
                    AIModelOption(
                        id=model_id,
                        label=f"{model_id}  (vision, tools)",
                        provider="ollama",
                    )
                )
        return sorted(options, key=lambda option: option.id.lower())

    def _get_ollama_capabilities(self, model_id: str) -> set:
        body = json.dumps({"model": model_id}).encode("utf-8")
        req = urllib.request.Request(
            f"{self._ollama_base_url}/api/show",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, json.JSONDecodeError):
            return set()

        capabilities = payload.get("capabilities") or []
        if isinstance(capabilities, list):
            return {str(item).lower() for item in capabilities}
        return set()

    def list_models(self) -> List[AIModelOption]:
        """Return model options for the current provider."""
        if self._provider == "ollama":
            return self.list_ollama_models()
        if self._provider == "anthropic":
            from pmg_qt.ai_provider_claude import list_models
            return list_models(self._get_client())
        return self.list_openai_models()

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _tool_run_command(self, command: str) -> str:
        """Execute a PyMOL command and capture any stdout/stderr output."""
        import sys
        from io import StringIO

        # Block interactive commands that would hang the thread indefinitely
        parts = _split_commands(command)
        for part in parts:
            token = _first_token(part).lower()
            if token in _BLOCKED_COMMANDS:
                return (
                    f"ERROR: '{token}' is blocked — it opens an interactive session "
                    "that hangs forever. Use direct PyMOL commands instead."
                )

        reason = self._confirmation_reason(parts)
        if reason and not (self._on_confirm and self._on_confirm(command, reason)):
            return f"DECLINED: the user did not approve this command ({reason}). It was not run."

        for part in parts:
            self._record_command(part)

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = out_buf = StringIO()
        sys.stderr = err_buf = StringIO()
        try:
            # Separate lines stop iterate/alter/label from swallowing later commands.
            self.cmd.do(parts if len(parts) > 1 else command, echo=1)
            stdout = out_buf.getvalue().strip()
            stderr = err_buf.getvalue().strip()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        parts = []
        if stdout:
            parts.append(stdout)
        if stderr:
            parts.append(f"[stderr] {stderr}")
        return "\n".join(parts) if parts else f"OK: {command}"

    def _resolve_keyword(self, token: str) -> Optional[str]:
        """Resolve a command word or abbreviation; None if PyMOL would run it as Python."""
        keyword = getattr(self.cmd, "keyword", None)
        kwhash = getattr(self.cmd, "kwhash", None)
        if keyword is None or kwhash is None or token in keyword:
            return token
        if token in kwhash:
            resolved = kwhash.interpret(token)
            if resolved is None:
                return kwhash[token]
            # An ambiguous abbreviation is an error in PyMOL's parser, not Python.
            return resolved if isinstance(resolved, str) else token
        return None

    def _confirmation_reason(self, parts: List[str]) -> str:
        """Return why these commands need the user's approval, or '' if they don't."""
        for part in parts:
            if part.startswith("/"):
                return "it runs Python code"
            token = _first_token(part)
            if token.startswith("@"):
                return "it runs a script file"
            resolved = self._resolve_keyword(token)
            if resolved is None:
                return "it is not a PyMOL command, so PyMOL would run it as Python"
            if resolved in _ALWAYS_CONFIRM_COMMANDS:
                return f"'{resolved}' can discard work or run other code"
            if resolved in ("delete", "del") and not self._deletes_only_selections(part, token):
                return "it deletes loaded objects"
            if resolved in _FILE_WRITING_COMMANDS:
                path = os.path.expanduser(_first_argument(part, token))
                if path and (os.path.exists(path) or os.path.exists(path + ".png")):
                    return f"it overwrites the existing file {path}"
        return ""

    def _deletes_only_selections(self, part: str, token: str) -> bool:
        names = part[len(token):].replace(",", " ").split()
        try:
            selections = self.cmd.get_names("selections")
        except Exception:  # noqa: BLE001
            return False
        return bool(names) and all(name in selections for name in names)

    def _record_command(self, command: str):
        self.last_turn_ran_commands = True
        if self._current_log is not None:
            self._current_log["commands"].append(command)

    def _tool_view_screen(self) -> str:
        """Capture the viewport as PNG and return a special image marker."""
        tmp = tempfile.mktemp(suffix=".png")
        try:
            self.cmd.png(tmp, quiet=1)
            # Give PyMOL a moment to flush the file
            if not os.path.exists(tmp):
                return "Error: Screenshot file was not created."
            self._downscale_screenshot(tmp)
            with open(tmp, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            return f"{_IMAGE_PREFIX}{b64}"
        finally:
            try:
                if os.path.exists(tmp):
                    os.unlink(tmp)
            except OSError:
                pass

    @staticmethod
    def _downscale_screenshot(path: str):
        """Shrink a captured PNG in place; on any failure the original is kept."""
        try:
            from pymol.Qt import QtCore, QtGui
            image = QtGui.QImage(path)
            if image.isNull() or max(image.width(), image.height()) <= _MAX_SCREENSHOT_EDGE:
                return
            image.scaled(
                _MAX_SCREENSHOT_EDGE,
                _MAX_SCREENSHOT_EDGE,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            ).save(path, "PNG")
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _http_json(url: str, payload: Optional[dict] = None, timeout: int = 15):
        """GET (or POST `payload` to) a JSON endpoint and return the decoded body."""
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        req = urllib.request.Request(url, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _rcsb_entry_details(self, pdb_ids: List[str]) -> Dict[str, str]:
        """Return a one-line description per PDB id; empty if the lookup fails."""
        query = (
            "query($ids:[String!]!){entries(entry_ids:$ids){rcsb_id struct{title} "
            "exptl{method} rcsb_entry_info{resolution_combined} "
            "nonpolymer_entities{pdbx_entity_nonpoly{comp_id}}}}"
        )
        try:
            result = self._http_json(
                "https://data.rcsb.org/graphql",
                {"query": query, "variables": {"ids": pdb_ids}},
            )
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, OSError):
            return {}

        details = {}
        for entry in (result.get("data") or {}).get("entries") or []:
            parts = [((entry.get("struct") or {}).get("title") or "").strip()]
            method = ", ".join(e.get("method", "") for e in entry.get("exptl") or [])
            resolution = (entry.get("rcsb_entry_info") or {}).get("resolution_combined")
            if method:
                parts.append(method + (f" {resolution[0]:.2f} A" if resolution else ""))
            ligands = [
                (item.get("pdbx_entity_nonpoly") or {}).get("comp_id")
                for item in entry.get("nonpolymer_entities") or []
            ]
            if ligands:
                parts.append("ligands: " + ", ".join(filter(None, ligands)))
            details[entry.get("rcsb_id", "")] = " | ".join(filter(None, parts))
        return details

    def _tool_search_pdb(self, query: str, max_results: int = 10) -> str:
        """Search RCSB PDB via the full-text search API and return matching entries."""
        max_results = min(int(max_results), 25)

        payload = {
            "query": {
                "type": "terminal",
                "service": "full_text",
                "parameters": {"value": query},
            },
            "return_type": "entry",
            "request_options": {
                "paginate": {"start": 0, "rows": max_results},
                "results_content_type": ["experimental"],
                "sort": [{"sort_by": "score", "direction": "desc"}],
                "scoring_strategy": "combined",
            },
        }

        data = json.dumps(payload).encode("utf-8")
        url = "https://search.rcsb.org/rcsbsearch/v2/query"
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read().decode("utf-8").strip()
                if not raw:
                    return f"No results found for '{query}'."
                result = json.loads(raw)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace").strip()
            if not body:
                return f"No results found for '{query}'."
            return f"PDB search error {e.code}: {body[:300]}"
        except urllib.error.URLError as e:
            return f"PDB search network error: {e.reason}"
        except json.JSONDecodeError as e:
            return f"PDB search returned unexpected response: {e}"

        hits = result.get("result_set", [])
        total = result.get("total_count", 0)

        if not hits:
            return f"No results found for '{query}'."

        details = self._rcsb_entry_details([hit.get("identifier", "?") for hit in hits])
        lines = [f"Found {total} total results (showing {len(hits)}):\n"]
        for hit in hits:
            pdb_id = hit.get("identifier", "?")
            score = hit.get("score", 0)
            lines.append(f"  {pdb_id}  (score: {score:.3f})  {details.get(pdb_id, '')}".rstrip())

        lines.append(
            "\nUse run_command('fetch <PDB_ID>') to load any of these structures."
        )
        return "\n".join(lines)

    def _tool_get_scene_info(self, object_name: str = "") -> str:
        """Return loaded objects with their chains and residue counts."""
        cmd = self.cmd

        if object_name:
            names = [object_name] if object_name in cmd.get_names("objects") else []
            if not names:
                return f"Object '{object_name}' not found. Loaded objects: {cmd.get_names('objects')}"
        else:
            names = cmd.get_names("objects")

        if not names:
            return "No objects are currently loaded in PyMOL."

        lines = [f"Loaded objects ({len(names)} total):\n"]
        for name in names:
            obj_type = "unknown"
            try:
                # get_type returns e.g. "object:molecule", "object:map", etc.
                obj_type = cmd.get_type(name)
            except Exception:
                pass

            lines.append(f"  [{name}]  type: {obj_type}")

            if "molecule" in obj_type:
                try:
                    chains = cmd.get_chains(name)
                    if chains:
                        # For each chain get residue count
                        chain_info = []
                        for ch in chains:
                            sel = f"{name} and chain {ch}"
                            # count unique residues
                            residues = set()
                            cmd.iterate(sel, "residues.add((resi, resn))",
                                        space={"residues": residues})
                            chain_info.append(f"chain {ch} ({len(residues)} residues)")
                        lines.append(f"    chains: {', '.join(chain_info)}")
                    else:
                        # No chain IDs — count residues directly
                        residues = set()
                        cmd.iterate(name, "residues.add((resi, resn))",
                                    space={"residues": residues})
                        lines.append(f"    chains: (none)  {len(residues)} residues total")
                except Exception as e:
                    lines.append(f"    (could not read chains: {e})")

        return "\n".join(lines)

    def _tool_get_selection_info(self, selection: str, max_residues: int = 100) -> str:
        """Summarize the atoms in a selection."""
        atoms: List[tuple] = []
        self.cmd.iterate(
            selection,
            "atoms.append((model, chain, resi, resn, b, ss))",
            space={"atoms": atoms},
        )
        if not atoms:
            return f"Selection '{selection}' matches no atoms."

        residues: Dict[tuple, str] = {}
        for model, chain, resi, resn, _b, ss in atoms:
            residues[(model, chain, resi, resn)] = ss
        bfactors = [atom[4] for atom in atoms]
        composition: Dict[str, int] = {}
        for (_model, _chain, _resi, resn) in residues:
            composition[resn] = composition.get(resn, 0) + 1
        structure = {"H": 0, "S": 0, "L": 0}
        for ss in residues.values():
            structure[ss if ss in structure else "L"] += 1

        lines = [
            f"Selection: {selection}",
            f"Atoms: {len(atoms)}   Residues: {len(residues)}",
            f"Objects: {', '.join(sorted({r[0] for r in residues}))}",
            f"Chains: {', '.join(sorted({r[1] or '(none)' for r in residues}))}",
            f"Secondary structure (residues): helix {structure['H']}, "
            f"sheet {structure['S']}, loop {structure['L']}",
            f"B-factor: min {min(bfactors):.2f}, mean {sum(bfactors) / len(bfactors):.2f}, "
            f"max {max(bfactors):.2f}",
            "Composition: " + ", ".join(
                f"{name} {count}" for name, count in
                sorted(composition.items(), key=lambda kv: (-kv[1], kv[0]))
            ),
        ]
        max_residues = max(0, int(max_residues))
        listed = list(residues)[:max_residues]
        lines.append("Residues: " + ", ".join(
            f"{model}/{chain}/{resn}`{resi}" for model, chain, resi, resn in listed
        ))
        if len(residues) > len(listed):
            lines.append(f"(+{len(residues) - len(listed)} more residues not listed)")
        return "\n".join(lines)

    def _single_atom_error(self, selections: List[str]) -> str:
        for selection in selections:
            count = self.cmd.count_atoms(selection)
            if count != 1:
                return f"Error: '{selection}' matches {count} atoms; it must match exactly 1."
        return ""

    def _tool_measure(self, arguments: dict) -> str:
        """Measure distances, angles, dihedrals or contacts without changing the scene."""
        kind = arguments.get("kind", "")
        selections = [arguments.get(f"selection{i}", "") for i in range(1, 5)]
        needed = {"distance": 2, "angle": 3, "dihedral": 4, "contacts": 2}.get(kind)
        if needed is None:
            return f"Unknown measurement kind: {kind}"
        if not all(selections[:needed]):
            return f"Error: '{kind}' needs selection1 to selection{needed}."

        if kind == "contacts":
            return self._measure_contacts(
                selections[0],
                selections[1],
                float(arguments.get("cutoff", 4.0)),
                bool(arguments.get("polar_only", False)),
            )

        error = self._single_atom_error(selections[:needed])
        if error:
            return error
        if kind == "distance":
            return f"Distance: {self.cmd.get_distance(*selections[:2]):.3f} A"
        if kind == "angle":
            return f"Angle: {self.cmd.get_angle(*selections[:3]):.2f} degrees"
        return f"Dihedral: {self.cmd.get_dihedral(*selections[:4]):.2f} degrees"

    def _measure_contacts(self, sel1: str, sel2: str, cutoff: float, polar_only: bool) -> str:
        pairs = self.cmd.find_pairs(sel1, sel2, cutoff=cutoff, mode=1 if polar_only else 0)
        if not pairs:
            return f"No contacts within {cutoff} A."

        contacts = []
        for (model1, index1), (model2, index2) in pairs:
            atom1 = f"{model1} and index {index1}"
            atom2 = f"{model2} and index {index2}"
            contacts.append((self.cmd.get_distance(atom1, atom2), atom1, atom2))
        contacts.sort()

        labels: Dict[tuple, str] = {}
        shown = contacts[:_MAX_CONTACTS_LISTED]
        for _distance, atom1, atom2 in shown:
            for atom in (atom1, atom2):
                found: List[str] = []
                self.cmd.iterate(
                    atom,
                    "found.append(f'{model}/{chain}/{resn}`{resi}/{name}')",
                    space={"found": found},
                )
                labels[atom] = found[0] if found else atom

        lines = [f"{len(contacts)} contact(s) within {cutoff} A (closest first):"]
        lines += [f"  {labels[a]} -- {labels[b]}: {d:.2f} A" for d, a, b in shown]
        if len(contacts) > len(shown):
            lines.append(f"(+{len(contacts) - len(shown)} more not listed)")
        return "\n".join(lines)

    def _tool_search_uniprot(self, query: str, max_results: int = 10) -> str:
        """Search UniProtKB and return accessions with names and organisms."""
        params = urllib.parse.urlencode({
            "query": query,
            "format": "json",
            "size": min(int(max_results), 25),
            "fields": "accession,id,protein_name,organism_name,length,reviewed",
        })
        try:
            result = self._http_json(f"https://rest.uniprot.org/uniprotkb/search?{params}")
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, OSError) as e:
            return f"UniProt search error: {e}"

        entries = result.get("results") or []
        if not entries:
            return f"No UniProt results for '{query}'."
        lines = [f"UniProt results for '{query}':"]
        for entry in entries:
            name = (
                ((entry.get("proteinDescription") or {}).get("recommendedName") or {})
                .get("fullName", {})
                .get("value", "(unnamed)")
            )
            organism = (entry.get("organism") or {}).get("scientificName", "?")
            length = (entry.get("sequence") or {}).get("length", "?")
            reviewed = "unreviewed" if "unreviewed" in entry.get("entryType", "").lower() else "reviewed"
            lines.append(
                f"  {entry.get('primaryAccession', '?')}  {entry.get('uniProtkbId', '')}  "
                f"{name} | {organism} | {length} aa | {reviewed}"
            )
        return "\n".join(lines)

    def _tool_fetch_alphafold(self, accession: str) -> str:
        """Load an AlphaFold DB model for a UniProt accession."""
        accession = (accession or "").strip().upper()
        if not re.fullmatch(r"[A-Z0-9]{6,10}", accession):
            return f"Error: '{accession}' is not a UniProt accession."
        try:
            entries = self._http_json(f"https://alphafold.ebi.ac.uk/api/prediction/{accession}")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return f"AlphaFold DB has no model for {accession}."
            return f"AlphaFold DB error {e.code}."
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, OSError) as e:
            return f"AlphaFold DB network error: {e}"

        if not entries:
            return f"AlphaFold DB has no model for {accession}."
        entry = entries[0]
        url = entry.get("cifUrl", "")
        if not url.startswith("https://alphafold.ebi.ac.uk/files/"):
            return f"AlphaFold DB returned an unexpected file location: {url}"

        name = entry.get("entryId") or f"AF-{accession}"
        self.cmd.load(url, name, quiet=1)
        self._record_command(f"load {url}, {name}")
        return (
            f"Loaded {name}: {entry.get('uniprotDescription', '')} "
            f"({entry.get('organismScientificName', '')}, residues "
            f"{entry.get('uniprotStart')}-{entry.get('uniprotEnd')}, "
            f"mean pLDDT {entry.get('globalMetricValue')}). "
            "B-factors hold per-residue pLDDT."
        )

    def _load_docs(self) -> list:
        """Load the docs file lines, cached on the instance."""
        if not hasattr(self, "_docs_lines"):
            if not os.path.exists(_DOCS_PATH):
                self._docs_lines = []
            else:
                with open(_DOCS_PATH, encoding="utf-8") as f:
                    self._docs_lines = f.readlines()
        return self._docs_lines

    def _tool_docs_grep(self, pattern: str, context_lines: int = 2) -> str:
        """Search docs for pattern, return matching lines with context and line numbers."""
        lines = self._load_docs()
        if not lines:
            return f"Documentation file not found at: {_DOCS_PATH}"

        context_lines = min(int(context_lines), 10)
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return f"Invalid regex pattern: {e}"

        matches = [i for i, line in enumerate(lines) if rx.search(line)]
        if not matches:
            return f"No matches found for '{pattern}' in PyMOL documentation."

        # Merge overlapping context windows
        results = []
        groups = []
        current = None
        for idx in matches:
            lo = max(0, idx - context_lines)
            hi = min(len(lines) - 1, idx + context_lines)
            if current is None:
                current = [lo, hi, [idx]]
            elif lo <= current[1] + 1:
                current[1] = max(current[1], hi)
                current[2].append(idx)
            else:
                groups.append(current)
                current = [lo, hi, [idx]]
        if current:
            groups.append(current)

        MAX_GROUPS = 10
        if len(groups) > MAX_GROUPS:
            results.append(f"Found {len(matches)} matches across {len(groups)} locations (showing first {MAX_GROUPS}):\n")
            groups = groups[:MAX_GROUPS]
        else:
            results.append(f"Found {len(matches)} match(es):\n")

        for lo, hi, match_idxs in groups:
            match_set = set(match_idxs)
            results.append(f"  --- lines {lo+1}-{hi+1} ---")
            for i in range(lo, hi + 1):
                prefix = ">>>" if i in match_set else "   "
                results.append(f"  {prefix} {i+1:4d}: {lines[i].rstrip()}")
            results.append("")

        return "\n".join(results)

    def _tool_docs_view(self, start_line: int, end_line: int) -> str:
        """Return a range of lines from the docs (1-indexed, inclusive)."""
        lines = self._load_docs()
        if not lines:
            return f"Documentation file not found at: {_DOCS_PATH}"

        total = len(lines)
        start_line = max(1, int(start_line))
        end_line = min(total, int(end_line))

        if start_line > end_line:
            return f"Invalid range: start_line ({start_line}) > end_line ({end_line})."
        if end_line - start_line > 150:
            end_line = start_line + 149  # cap to prevent huge dumps

        result = [f"PyMOL docs lines {start_line}-{end_line} (of {total}):\n"]
        for i in range(start_line - 1, end_line):
            result.append(f"  {i+1:4d}: {lines[i].rstrip()}")
        return "\n".join(result)

    def _execute_tool(self, tool_name: str, arguments: dict) -> str:
        try:
            result = self._dispatch_tool(tool_name, arguments)
        except Exception as e:  # noqa: BLE001
            return f"Error running {tool_name}: {e}"
        if len(result) > _MAX_TOOL_OUTPUT_CHARS and not result.startswith(_IMAGE_PREFIX):
            return result[:_MAX_TOOL_OUTPUT_CHARS] + "\n[Output truncated.]"
        return result

    def _dispatch_tool(self, tool_name: str, arguments: dict) -> str:
        if tool_name == "run_command":
            return self._tool_run_command(arguments.get("command", ""))
        elif tool_name == "view_screen":
            return self._tool_view_screen()
        elif tool_name == "get_scene_info":
            return self._tool_get_scene_info(arguments.get("object_name", ""))
        elif tool_name == "get_selection_info":
            return self._tool_get_selection_info(
                arguments.get("selection", "all"),
                arguments.get("max_residues", 100),
            )
        elif tool_name == "measure":
            return self._tool_measure(arguments)
        elif tool_name == "search_pdb":
            return self._tool_search_pdb(
                arguments.get("query", ""),
                arguments.get("max_results", 10),
            )
        elif tool_name == "search_uniprot":
            return self._tool_search_uniprot(
                arguments.get("query", ""),
                arguments.get("max_results", 10),
            )
        elif tool_name == "fetch_alphafold":
            return self._tool_fetch_alphafold(arguments.get("accession", ""))
        elif tool_name == "docs_grep":
            return self._tool_docs_grep(
                arguments.get("pattern", ""),
                arguments.get("context_lines", 2),
            )
        elif tool_name == "docs_view":
            return self._tool_docs_view(
                arguments.get("start_line", 1),
                arguments.get("end_line", 50),
            )
        return f"Unknown tool: {tool_name}"

    @staticmethod
    def _describe_tool_call(tool_name: str, args: dict) -> str:
        if tool_name == "run_command":
            return f"Running PyMOL command: {args.get('command', '')}"
        if tool_name == "view_screen":
            return "Capturing viewport screenshot..."
        if tool_name == "get_scene_info":
            target = args.get("object_name", "")
            return f"Inspecting object: {target}" if target else "Listing loaded objects..."
        if tool_name == "get_selection_info":
            return f"Inspecting selection: {args.get('selection', '')}"
        if tool_name == "measure":
            return f"Measuring {args.get('kind', '')}: {args.get('selection1', '')} / {args.get('selection2', '')}"
        if tool_name == "search_pdb":
            return f"Searching PDB: {args.get('query', '')}"
        if tool_name == "search_uniprot":
            return f"Searching UniProt: {args.get('query', '')}"
        if tool_name == "fetch_alphafold":
            return f"Fetching AlphaFold model: {args.get('accession', '')}"
        if tool_name == "docs_grep":
            return f"Searching docs: {args.get('pattern', '')}"
        if tool_name == "docs_view":
            return f"Reading docs lines {args.get('start_line')}–{args.get('end_line')}"
        return f"Running: {tool_name}"

    def _scene_context(self) -> str:
        """Describe loaded objects and selections for the start of a user message."""
        try:
            objects = self.cmd.get_names("public_objects")
            selections = self.cmd.get_names("public_selections")
            counted = [f"{name} ({self.cmd.count_atoms(name)} atoms)" for name in selections[:20]]
        except Exception:  # noqa: BLE001
            return ""
        return (
            f"[PyMOL state] objects: {', '.join(objects[:40]) or 'none'}; "
            f"selections: {', '.join(counted) or 'none'}"
        )

    # ------------------------------------------------------------------
    # Agentic loop
    # ------------------------------------------------------------------

    def _raise_if_cancelled(self):
        if self._cancelled:
            raise InterruptedError("Cancelled by user.")

    def _commit_turn(self, adapter, turn: List[Any]):
        self._turns.append(turn)
        if adapter.trims_history:
            self._turns = self._turns[-_MAX_HISTORY_TURNS:]

    def _close_interrupted_turn(self, adapter, turn: List[Any], note: str):
        """Keep whole tool rounds plus a note saying what already ran."""
        commands = (self._current_log or {}).get("commands") or []
        if commands:
            note += " Commands already run: " + "; ".join(commands)
        self._commit_turn(adapter, turn + [adapter.assistant_text_item(f"[{note}]")])

    def send_message(
        self,
        user_text: str,
        on_tool_call: Optional[Callable[[str], None]] = None,
        on_reasoning: Optional[Callable[[str], None]] = None,
        on_text: Optional[Callable[[str], None]] = None,
        on_tool_result: Optional[Callable[[str], None]] = None,
        on_confirm: Optional[Callable[[str, str], bool]] = None,
        on_usage: Optional[Callable[[int, int], None]] = None,
    ) -> str:
        """
        Send a user message, run the agentic tool-use loop, and return the
        final assistant text.

        Conversation history is preserved across calls.

        Args:
            user_text: The user's chat message.
            on_tool_call: Called with a human-readable description of each tool call.
            on_reasoning: Called with reasoning summary text as it arrives.
            on_text: Called with assistant text as it arrives.
            on_tool_result: Called with the text output of each tool call.
            on_confirm: Called with (command, reason) before a risky command runs;
                        the command is skipped unless it returns True.
            on_usage: Called with (input_tokens, output_tokens) after each request.

        Returns:
            The final assistant response text.
        """
        client = self._get_client()
        adapter = self._adapter()

        self._cancelled = False
        self._on_confirm = on_confirm
        self.last_turn_ran_commands = False
        self._current_log = {"prompt": user_text, "commands": []}
        self._command_log.append(self._current_log)

        callbacks = StreamCallbacks(
            on_text=on_text or (lambda _text: None),
            on_reasoning=on_reasoning or (lambda _text: None),
            check_cancelled=self._raise_if_cancelled,
        )

        context = self._scene_context()
        # Only complete rounds join `turn`, so history never holds an unanswered call.
        turn: List[Any] = [
            adapter.user_item(f"{context}\n\n{user_text}" if context else user_text)
        ]

        try:
            for _ in range(_MAX_TOOL_ROUNDS):
                self._raise_if_cancelled()
                request_items = self._messages + turn
                adapter.prune_screenshots(request_items)
                result = adapter.create(
                    client, self._model, self._reasoning_effort, request_items, callbacks
                )
                if on_usage:
                    on_usage(result.input_tokens, result.output_tokens)

                if result.problem:
                    self._close_interrupted_turn(adapter, turn, result.problem)
                    return f"{result.text}\n\n{result.problem}".strip()

                if not result.tool_calls:
                    self._commit_turn(adapter, turn + result.items)
                    return result.text

                outputs = []
                for call in result.tool_calls:
                    if on_tool_call:
                        on_tool_call(self._describe_tool_call(call.name, call.arguments))
                    output = self._execute_tool(call.name, call.arguments)
                    self._raise_if_cancelled()

                    # Screenshots travel as an image alongside their text output.
                    if output.startswith(_IMAGE_PREFIX):
                        text = "Screenshot captured. See the attached image."
                        outputs.append((call, text, output[len(_IMAGE_PREFIX):]))
                    else:
                        text = output
                        outputs.append((call, text, None))
                    if on_tool_result:
                        on_tool_result(text)

                turn += result.items
                turn += adapter.tool_result_items(outputs)

            note = (
                f"Stopped after {_MAX_TOOL_ROUNDS} rounds of tool calls without "
                "finishing. Ask me to continue if you want me to keep going."
            )
            self._close_interrupted_turn(adapter, turn, note)
            return note
        except Exception as exc:
            # A failure before anything ran can simply be retried by the user.
            if len(turn) > 1 or self._current_log["commands"]:
                reason = "Cancelled by the user" if isinstance(exc, InterruptedError) else "Failed"
                self._close_interrupted_turn(adapter, turn, f"{reason} before finishing.")
            raise
        finally:
            if not self._current_log["commands"]:
                self._command_log.remove(self._current_log)
            self._current_log = None
            self._on_confirm = None

    def revert_last_turn(self):
        """Forget the last turn's commands and tell the model the scene was restored."""
        if self._command_log:
            self._command_log.pop()
        self.last_turn_ran_commands = False
        adapter = self._adapter()
        note = "[The user undid your previous turn. The scene is back to how it was before that turn.]"
        self._commit_turn(adapter, [adapter.user_item(note)])

    def export_script(self) -> str:
        """Return the commands run so far as a PyMOL .pml script."""
        lines = ["# PyMOL script exported from the AI Assistant", ""]
        for entry in self._command_log:
            lines.append("# " + " ".join(entry["prompt"].split()))
            lines.extend(entry["commands"])
            lines.append("")
        return "\n".join(lines)

    def has_commands(self) -> bool:
        return bool(self._command_log)

    def clear_command_log(self):
        self._command_log = []

    def set_reasoning_effort(self, effort: str):
        """Set reasoning effort: 'high', 'medium', 'low', or 'none'."""
        self._reasoning_effort = effort

    def get_reasoning_effort(self) -> str:
        return self._reasoning_effort

    def is_cancelled(self) -> bool:
        return self._cancelled

    def cancel(self):
        """Signal the agentic loop to stop after the current operation."""
        self._cancelled = True

    def clear_history(self):
        """Clear the conversation history."""
        self._cancelled = False
        self._turns = []
