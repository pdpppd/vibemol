import json
from types import SimpleNamespace

import pytest

from pmg_qt.ai_agent import _MAX_TOOL_ROUNDS, _ResponsesAdapter, AIAgent, ToolCall


class DummyCmd:
    def __init__(self):
        self.commands = []

    def do(self, command, echo=1):
        self.commands.append((command, echo))


class FakeResponse:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self._payload


def test_ollama_model_discovery_filters_for_vision_and_tools(monkeypatch):
    capability_map = {
        "ok:latest": ["completion", "vision", "tools"],
        "text-only:latest": ["completion", "tools"],
        "vision-only:latest": ["completion", "vision"],
    }

    def fake_urlopen(request, timeout):
        url = getattr(request, "full_url", request)
        if url.endswith("/api/tags"):
            return FakeResponse({"models": [{"model": name} for name in capability_map]})

        body = json.loads(request.data.decode("utf-8"))
        return FakeResponse({"capabilities": capability_map[body["model"]]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    agent = AIAgent(DummyCmd(), provider="ollama")
    options = agent.list_models()

    assert [option.id for option in options] == ["ok:latest"]


def test_openai_model_discovery_uses_available_models(monkeypatch):
    agent = AIAgent(DummyCmd(), api_key="test-key")
    fake_client = SimpleNamespace(
        models=SimpleNamespace(
            list=lambda: [
                SimpleNamespace(id="custom-model"),
                SimpleNamespace(id="gpt-5.4"),
            ]
        )
    )
    monkeypatch.setattr(agent, "_get_client", lambda: fake_client)

    options = agent.list_models()

    assert [option.id for option in options] == ["gpt-5.4", "custom-model"]


def test_local_agent_loop_uses_tools_without_reasoning_payload():
    responses = []

    class FakeResponses:
        def create(self, **kwargs):
            responses.append(kwargs)
            if len(responses) == 1:
                return SimpleNamespace(
                    output=[
                        SimpleNamespace(
                            type="function_call",
                            name="run_command",
                            arguments=json.dumps({"command": "get_names"}),
                            call_id="call-1",
                        )
                    ],
                    output_text="",
                )
            return SimpleNamespace(output=[], output_text="done")

    cmd = DummyCmd()
    agent = AIAgent(cmd, provider="ollama", model="ok:latest")
    agent._client = SimpleNamespace(responses=FakeResponses())

    assert agent.send_message("what is loaded?") == "done"
    assert cmd.commands == [("get_names", 1)]
    assert "reasoning" not in responses[0]
    assert responses[0]["model"] == "ok:latest"
    assert responses[0]["tools"]
    assert any(
        item.get("type") == "function_call_output"
        for item in responses[1]["input"]
        if isinstance(item, dict)
    )


def function_call(name, arguments, call_id="call-1"):
    return SimpleNamespace(
        type="function_call", name=name, arguments=json.dumps(arguments), call_id=call_id
    )


class ScriptedResponses:
    """Fake Responses client that replays turns and records each request's input."""

    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.inputs = []

    def create(self, **kwargs):
        self.inputs.append(list(kwargs["input"]))
        output = self.outputs.pop(0)
        text = output if isinstance(output, str) else ""
        items = [] if isinstance(output, str) else output
        return SimpleNamespace(output=items, output_text=text)


def scripted_agent(*outputs, cmd=None):
    agent = AIAgent(cmd or DummyCmd(), provider="ollama", model="ok:latest")
    fake = ScriptedResponses(*outputs)
    agent._client = SimpleNamespace(responses=fake)
    return agent, fake


def test_tool_calls_stay_in_history_for_the_next_message():
    agent, fake = scripted_agent(
        [function_call("run_command", {"command": "color red, all"})],
        "colored",
        "yes",
    )

    agent.send_message("color it")
    agent.send_message("did you color it?")

    second_turn = fake.inputs[2]
    assert any(
        isinstance(item, dict) and item.get("type") == "function_call_output"
        for item in second_turn
    )
    assert any(getattr(item, "type", None) == "function_call" for item in second_turn)


def test_cancel_during_tool_leaves_no_unanswered_call_in_history():
    cmd = DummyCmd()
    agent, fake = scripted_agent(
        [function_call("run_command", {"command": "color red, all"})],
        "next answer",
        cmd=cmd,
    )
    original_do = cmd.do

    def cancel_after_running(command, echo=1):
        original_do(command, echo)
        agent.cancel()

    cmd.do = cancel_after_running

    with pytest.raises(InterruptedError):
        agent.send_message("color it")

    assert not any(getattr(item, "type", None) == "function_call" for item in agent._messages)
    assert "color red, all" in agent._messages[-1]["content"]
    assert agent.send_message("try again") == "next answer"


def test_loop_stops_at_the_round_limit():
    rounds = [[function_call("docs_view", {"start_line": 1, "end_line": 2})] for _ in range(_MAX_TOOL_ROUNDS)]
    agent, fake = scripted_agent(*rounds)

    reply = agent.send_message("loop forever")

    assert f"{_MAX_TOOL_ROUNDS} rounds" in reply
    assert len(fake.inputs) == _MAX_TOOL_ROUNDS
    outputs = [i for i in agent._messages if isinstance(i, dict) and i.get("type") == "function_call_output"]
    assert len(outputs) == _MAX_TOOL_ROUNDS


def test_only_recent_screenshots_are_sent(monkeypatch):
    agent, fake = scripted_agent(
        *[[function_call("view_screen", {})] for _ in range(5)],
        "done",
    )
    monkeypatch.setattr(agent, "_tool_view_screen", lambda: "__IMAGE__:abc")

    agent.send_message("look repeatedly")

    blocks = [
        block for item in fake.inputs[-1]
        if isinstance(item, dict) and isinstance(item.get("content"), list)
        for block in item["content"]
    ]
    assert sum(block["type"] == "input_image" for block in blocks) == 3
    assert sum(block["type"] == "input_text" for block in blocks) == 2


def test_streamed_text_and_reasoning_reach_callbacks():
    agent = AIAgent(DummyCmd(), api_key="test-key")
    final = SimpleNamespace(output=[], output_text="Hello", status="completed", usage=SimpleNamespace(input_tokens=7, output_tokens=3))

    class FakeStream(list):
        closed = False

        def close(self):
            FakeStream.closed = True

    def create(stream=False, **kwargs):
        assert stream
        return FakeStream([
            SimpleNamespace(type="response.reasoning_summary_part.added"),
            SimpleNamespace(type="response.reasoning_summary_text.delta", delta="hm"),
            SimpleNamespace(type="response.output_text.delta", delta="Hel"),
            SimpleNamespace(type="response.output_text.delta", delta="lo"),
            SimpleNamespace(type="response.completed", response=final),
        ])

    agent._client = SimpleNamespace(responses=SimpleNamespace(create=create))
    text, reasoning, usage = [], [], []

    reply = agent.send_message(
        "hi",
        on_text=text.append,
        on_reasoning=reasoning.append,
        on_usage=lambda used_in, used_out: usage.append((used_in, used_out)),
    )

    assert reply == "Hello"
    assert text == ["Hel", "lo"]
    assert reasoning == ["hm"]
    assert usage == [(7, 3)]
    assert FakeStream.closed


def test_incomplete_response_does_not_run_truncated_tool_call():
    cmd = DummyCmd()
    agent = AIAgent(cmd, api_key="test-key")
    agent._adapter = lambda: _ResponsesAdapter(sends_reasoning=False, streaming=False)
    response = SimpleNamespace(
        output=[function_call("run_command", {"command": "delete all"})],
        output_text="partial",
        status="incomplete",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
    )
    agent._client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: response))

    reply = agent.send_message("go")

    assert cmd.commands == []
    assert "cut off" in reply


class KeywordCmd(DummyCmd):
    keyword = {"delete": 1, "color": 1, "remove": 1, "save": 1}
    kwhash = {}

    def get_names(self, kind="objects"):
        return ["sele_a"] if kind == "selections" else []


@pytest.mark.parametrize(
    "command,allowed_without_asking",
    [
        ("color red, all", True),
        ("delete sele_a", True),
        ("delete all", False),
        ("color red; remove chain A", False),
        ("/import os", False),
        ("@script.pml", False),
        ("os.getcwd()", False),
    ],
)
def test_risky_commands_need_confirmation(command, allowed_without_asking):
    cmd = KeywordCmd()
    agent = AIAgent(cmd, api_key="test-key")

    result = agent._tool_run_command(command)

    assert bool(cmd.commands) == allowed_without_asking
    assert result.startswith("DECLINED") != allowed_without_asking


def test_approved_command_runs_and_interactive_ones_stay_blocked():
    cmd = KeywordCmd()
    agent = AIAgent(cmd, api_key="test-key")
    agent._on_confirm = lambda command, reason: True

    agent._tool_run_command("delete all")
    assert cmd.commands == [("delete all", 1)]

    assert "blocked" in agent._tool_run_command("color red; python")
    assert len(cmd.commands) == 1


def test_save_only_asks_when_the_file_exists(tmp_path):
    cmd = KeywordCmd()
    agent = AIAgent(cmd, api_key="test-key")
    existing = tmp_path / "old.pdb"
    existing.write_text("x")

    assert agent._tool_run_command(f"save {tmp_path / 'new.pdb'}, all").startswith("OK")
    assert agent._tool_run_command(f"save {existing}, all").startswith("DECLINED")


def test_export_and_undo_track_commands():
    agent, _fake = scripted_agent(
        [function_call("run_command", {"command": "color red, all"})], "one",
        [function_call("run_command", {"command": "color blue, all"})], "two",
    )
    agent.send_message("make it red")
    agent.send_message("make it blue")
    assert agent.last_turn_ran_commands

    script = agent.export_script()
    assert "# make it red\ncolor red, all" in script
    assert "color blue, all" in script

    agent.revert_last_turn()

    assert "color blue, all" not in agent.export_script()
    assert "undid" in agent._messages[-1]["content"]


def test_search_pdb_includes_entry_details(monkeypatch):
    agent = AIAgent(DummyCmd(), api_key="test-key")

    def fake_urlopen(request, timeout):
        if "graphql" in request.full_url:
            return FakeResponse({"data": {"entries": [{
                "rcsb_id": "1HPV",
                "struct": {"title": "HIV PROTEASE"},
                "exptl": [{"method": "X-RAY DIFFRACTION"}],
                "rcsb_entry_info": {"resolution_combined": [1.9]},
                "nonpolymer_entities": [{"pdbx_entity_nonpoly": {"comp_id": "478"}}],
            }]}})
        return FakeResponse({"total_count": 1, "result_set": [{"identifier": "1HPV", "score": 1.0}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = agent._tool_search_pdb("hiv")

    assert "HIV PROTEASE" in result
    assert "1.90 A" in result
    assert "ligands: 478" in result


def test_claude_adapter_formats_screenshots_and_keeps_history_unedited():
    pytest.importorskip("anthropic")
    from pmg_qt.ai_provider_claude import ClaudeAdapter

    adapter = ClaudeAdapter()
    call = ToolCall("tool-1", "view_screen", {})
    items = adapter.tool_result_items([(call, "shot", "abc")]) * 5
    snapshot = json.dumps(items)

    adapter.prune_screenshots(items)

    assert json.dumps(items) == snapshot
    block = items[0]["content"][0]
    assert block["tool_use_id"] == "tool-1"
    assert block["content"][0]["source"]["data"] == "abc"


def test_command_chains_run_as_separate_lines():
    cmd = DummyCmd()
    agent = AIAgent(cmd, api_key="test-key")
    agent._current_log = {"prompt": "p", "commands": []}

    agent._tool_run_command('label all, "a;b"; color red, all')

    assert cmd.commands == [(['label all, "a;b"', "color red, all"], 1)]
    assert agent._current_log["commands"] == ['label all, "a;b"', "color red, all"]
