"""
AI Agent for PyMOL - uses OpenAI Responses API to run an agentic loop
with tools for executing PyMOL commands and capturing the viewport.
"""

import base64
import json
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable

_IMAGE_PREFIX = "__IMAGE__:"
_DEFAULT_OPENAI_MODEL = "gpt-5.4"
_DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


@dataclass(frozen=True)
class AIModelOption:
    """A model option shown in the Qt model picker."""

    id: str
    label: str
    provider: str

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
- get_names             : list all loaded objects (or use get_scene_info() for richer detail)
- png <filename>        : save viewport image (used internally by view_screen)

## Finding structures:
Use search_pdb() to find PDB IDs by keyword before fetching. For example, if the user
asks for "a kinase with an inhibitor", search first to find relevant IDs, then fetch
the best match. Always tell the user which structure you chose and why.

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


class AIAgent:
    """
    Wraps the OpenAI Responses API in an agentic loop with PyMOL tools.

    Maintains full conversation history across multiple send_message() calls
    within a session.
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
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
            provider: "openai" or "ollama".
            model: Model id to use. Falls back to a provider-specific default.
            ollama_base_url: Base URL for the local Ollama server.
        """
        self.cmd = cmd
        self._messages: List[Dict[str, Any]] = []
        self._client = None
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._provider = provider
        self._model = model or _DEFAULT_OPENAI_MODEL
        self._ollama_base_url = ollama_base_url.rstrip("/")
        self._reasoning_effort = "high"  # "high" | "medium" | "low" | "none"
        self._cancelled = False

    def _get_client(self):
        if self._client is None:
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

            if not self._api_key:
                raise RuntimeError(
                    "No OpenAI API key found. Enter one in the AI panel or set "
                    "the OPENAI_API_KEY environment variable."
                )
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    # ------------------------------------------------------------------
    # Model discovery and configuration
    # ------------------------------------------------------------------

    def set_provider(self, provider: str):
        """Set the model provider: 'openai' or 'ollama'."""
        if provider not in ("openai", "ollama"):
            raise ValueError(f"Unknown AI provider: {provider}")
        if provider != self._provider:
            self._provider = provider
            self._client = None
            self.clear_history()

    def set_api_key(self, api_key: str):
        """Set the OpenAI API key used for future OpenAI requests."""
        api_key = (api_key or "").strip()
        if api_key != self._api_key:
            self._api_key = api_key
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
        return self.list_openai_models()

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _tool_run_command(self, command: str) -> str:
        """Execute a PyMOL command and capture any stdout/stderr output."""
        import sys
        from io import StringIO

        # Block interactive commands that would hang the thread indefinitely
        first_token = command.strip().split()[0].lower() if command.strip() else ""
        if first_token in ("python", "python3", "ipython", "shell", "system"):
            return (
                f"ERROR: '{first_token}' is blocked — it opens an interactive session "
                "that hangs forever. Use direct PyMOL commands instead."
            )

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = out_buf = StringIO()
        sys.stderr = err_buf = StringIO()
        try:
            self.cmd.do(command, echo=1)
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

    def _tool_view_screen(self) -> str:
        """Capture the viewport as PNG and return a special image marker."""
        tmp = tempfile.mktemp(suffix=".png")
        try:
            self.cmd.png(tmp, quiet=1)
            # Give PyMOL a moment to flush the file
            if not os.path.exists(tmp):
                return "Error: Screenshot file was not created."
            with open(tmp, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            return f"{_IMAGE_PREFIX}{b64}"
        finally:
            try:
                if os.path.exists(tmp):
                    os.unlink(tmp)
            except OSError:
                pass

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

        lines = [f"Found {total} total results (showing {len(hits)}):\n"]
        for hit in hits:
            pdb_id = hit.get("identifier", "?")
            score = hit.get("score", 0)
            lines.append(f"  {pdb_id}  (score: {score:.3f})")

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
        import re
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
        if tool_name == "run_command":
            return self._tool_run_command(arguments.get("command", ""))
        elif tool_name == "view_screen":
            return self._tool_view_screen()
        elif tool_name == "get_scene_info":
            return self._tool_get_scene_info(arguments.get("object_name", ""))
        elif tool_name == "search_pdb":
            return self._tool_search_pdb(
                arguments.get("query", ""),
                arguments.get("max_results", 10),
            )
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

    # ------------------------------------------------------------------
    # Agentic loop
    # ------------------------------------------------------------------

    def send_message(
        self,
        user_text: str,
        on_tool_call: Optional[Callable[[str], None]] = None,
        on_reasoning: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Send a user message, run the agentic tool-use loop, and return the
        final assistant text.

        Conversation history is preserved across calls.

        Args:
            user_text: The user's chat message.
            on_tool_call: Optional callback called with a human-readable
                          description each time a tool is invoked.

        Returns:
            The final assistant response text.
        """
        client = self._get_client()

        self._cancelled = False
        # input starts as the full conversation history plus the new user message.
        # After each round-trip we append the model's output items and our tool
        # results so the next call sees the complete context.
        input_messages: List[Any] = list(self._messages)
        input_messages.append({"role": "user", "content": user_text})

        while True:
            if self._cancelled:
                raise InterruptedError("Cancelled by user.")

            request = {
                "model": self._model,
                "instructions": SYSTEM_PROMPT,
                "tools": TOOLS,
                "input": input_messages,
            }
            if self._provider == "openai" and self._reasoning_effort != "none":
                request["reasoning"] = {
                    "effort": self._reasoning_effort,
                    "summary": "auto",
                }

            response = client.responses.create(**request)

            # Emit reasoning summaries before processing tool calls
            if on_reasoning:
                for item in response.output:
                    if item.type == "reasoning":
                        summaries = getattr(item, "summary", None) or []
                        for s in summaries:
                            text = getattr(s, "text", None) or str(s)
                            if text:
                                on_reasoning(text)

            tool_calls = [item for item in response.output if item.type == "function_call"]

            if not tool_calls:
                # Final answer — persist the full exchange into long-term history
                assistant_text = response.output_text or ""
                self._messages.append({"role": "user", "content": user_text})
                self._messages.append({"role": "assistant", "content": assistant_text})
                return assistant_text

            # Append model output items to the running input for the next round-trip.
            # The SDK objects are accepted directly by the Responses API as input.
            input_messages += response.output

            # Execute each tool call and append results
            for tool_call in tool_calls:
                tool_name = tool_call.name
                try:
                    args = json.loads(tool_call.arguments)
                except (json.JSONDecodeError, AttributeError):
                    args = {}

                if on_tool_call:
                    if tool_name == "run_command":
                        desc = f"Running PyMOL command: {args.get('command', '')}"
                    elif tool_name == "view_screen":
                        desc = "Capturing viewport screenshot..."
                    elif tool_name == "get_scene_info":
                        target = args.get("object_name", "")
                        desc = f"Inspecting object: {target}" if target else "Listing loaded objects..."
                    elif tool_name == "search_pdb":
                        desc = f"Searching PDB: {args.get('query', '')}"
                    elif tool_name == "docs_grep":
                        desc = f"Searching docs: {args.get('pattern', '')}"
                    elif tool_name == "docs_view":
                        desc = f"Reading docs lines {args.get('start_line')}–{args.get('end_line')}"
                    else:
                        desc = f"Running: {tool_name}"
                    on_tool_call(desc)

                result = self._execute_tool(tool_name, args)

                if self._cancelled:
                    raise InterruptedError("Cancelled by user.")

                # Special handling for screenshot: inject an image message
                # so the model can actually see the pixel content
                if result.startswith(_IMAGE_PREFIX):
                    b64_data = result[len(_IMAGE_PREFIX):]
                    input_messages.append({
                        "type": "function_call_output",
                        "call_id": tool_call.call_id,
                        "output": "Screenshot captured. See the attached image.",
                    })
                    input_messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": f"data:image/png;base64,{b64_data}",
                            }
                        ],
                    })
                else:
                    input_messages.append({
                        "type": "function_call_output",
                        "call_id": tool_call.call_id,
                        "output": result,
                    })

    def set_reasoning_effort(self, effort: str):
        """Set reasoning effort: 'high', 'medium', 'low', or 'none'."""
        self._reasoning_effort = effort

    def cancel(self):
        """Signal the agentic loop to stop after the current operation."""
        self._cancelled = True

    def clear_history(self):
        """Clear the conversation history."""
        self._cancelled = False
        self._messages = []
