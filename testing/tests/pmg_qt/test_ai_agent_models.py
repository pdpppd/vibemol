import json
from types import SimpleNamespace

from pmg_qt.ai_agent import AIAgent


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
