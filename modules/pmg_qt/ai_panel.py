"""
AI Chat Panel for PyMOL.

Provides a dockable QDockWidget with a chat interface that talks to an LLM
agent capable of running PyMOL commands and capturing the viewport.
"""

import threading

from pymol.Qt import QtWidgets, QtCore, QtGui

Qt = QtCore.Qt

_KEYRING_SERVICE = "PyMOL AI Assistant"
_MAX_ACTION_OUTPUT_CHARS = 600
_PRESET_PROMPTS = (
    (
        "Binding site",
        "Show the ligand binding site: ligand as sticks, residues within 4 A as sticks, "
        "polar contacts as dashes, and the rest of the protein as a faded cartoon.",
    ),
    (
        "B-factor",
        "Color the structure by B-factor and tell me which regions are the most flexible.",
    ),
    (
        "Figure",
        "Make this view publication quality: white background, clean cartoon, sensible "
        "colors and depth cues, and no distracting representations.",
    ),
)


def _settings():
    return QtCore.QSettings("PyMOL", "AIAssistant")


def _load_saved_api_key(provider: str) -> str:
    """Return a key from the system keychain, or '' if unavailable."""
    try:
        import keyring
        return keyring.get_password(_KEYRING_SERVICE, provider) or ""
    except Exception:  # noqa: BLE001
        return ""


def _save_api_key(provider: str, api_key: str) -> bool:
    """Store a key in the system keychain. Keys are never written in plain text."""
    try:
        import keyring
        if api_key:
            keyring.set_password(_KEYRING_SERVICE, provider, api_key)
        elif keyring.get_password(_KEYRING_SERVICE, provider):
            keyring.delete_password(_KEYRING_SERVICE, provider)
        return True
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# Worker thread for running the agent without blocking the UI
# ---------------------------------------------------------------------------

class _AgentWorker(QtCore.QThread):
    """Runs AIAgent.send_message() in a background thread."""

    finished = QtCore.Signal(str)       # emits final response text
    tool_called = QtCore.Signal(str)    # emits human-readable tool description
    tool_result = QtCore.Signal(str)    # emits a tool's text output
    reasoning = QtCore.Signal(str)      # emits model reasoning summary text
    text_delta = QtCore.Signal(str)     # emits assistant text as it arrives
    usage = QtCore.Signal(int, int)     # emits input and output token counts
    confirm_requested = QtCore.Signal(str, str)  # emits command and reason
    error = QtCore.Signal(str)          # emits error message

    def __init__(self, agent, user_text: str, parent=None):
        super().__init__(parent)
        self._agent = agent
        self._user_text = user_text
        self._confirm_event = threading.Event()
        self._confirm_allowed = False
        self.snapshot = None

    def run(self):
        try:
            self.snapshot = self._agent.cmd.get_session()
        except Exception:  # noqa: BLE001
            self.snapshot = None

        try:
            response = self._agent.send_message(
                self._user_text,
                on_tool_call=lambda desc: self.tool_called.emit(desc),
                on_reasoning=lambda text: self.reasoning.emit(text),
                on_text=lambda text: self.text_delta.emit(text),
                on_tool_result=lambda text: self.tool_result.emit(text),
                on_confirm=self._confirm,
                on_usage=lambda used_in, used_out: self.usage.emit(used_in, used_out),
            )
            self.finished.emit(response or "")
        except InterruptedError:
            self.finished.emit("")   # cancelled — emit empty, UI handles it
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))

    def _confirm(self, command: str, reason: str) -> bool:
        """Ask the GUI thread for approval and wait; Cancel releases the wait."""
        self._confirm_event.clear()
        self._confirm_allowed = False
        self.confirm_requested.emit(command, reason)
        while not self._confirm_event.wait(0.1):
            if self._agent.is_cancelled():
                return False
        return self._confirm_allowed

    def answer_confirmation(self, allowed: bool):
        self._confirm_allowed = allowed
        self._confirm_event.set()


class _ModelDiscoveryWorker(QtCore.QThread):
    """Loads model options without blocking the Qt UI."""

    finished = QtCore.Signal(object)    # emits list[AIModelOption]
    error = QtCore.Signal(str)

    def __init__(self, agent, parent=None):
        super().__init__(parent)
        self._agent = agent

    def run(self):
        try:
            self.finished.emit(self._agent.list_models())
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Collapsible action log widget
# ---------------------------------------------------------------------------

class _CollapsibleActions(QtWidgets.QFrame):
    """
    A collapsible widget that shows a summary header (e.g. "3 actions taken")
    and expands to reveal individual tool-call entries. Collapsed by default.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._entries = []

        self.setStyleSheet(
            "QFrame { background-color: #2a2a2a; border-radius: 6px; border: 1px solid #3d6b3d; }"
        )

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # --- Header button (always visible, acts as toggle) ---
        self._toggle_btn = QtWidgets.QPushButton()
        self._toggle_btn.setFlat(True)
        self._toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle_btn.setStyleSheet(
            "QPushButton {"
            "  color: #6dbf6d;"
            "  font-size: 11px;"
            "  text-align: left;"
            "  padding: 5px 8px;"
            "  border: none;"
            "  background: transparent;"
            "}"
            "QPushButton:hover { color: #90d990; }"
        )
        self._toggle_btn.clicked.connect(self._toggle)
        outer.addWidget(self._toggle_btn)

        # --- Body (hidden by default) ---
        self._body = QtWidgets.QWidget()
        self._body.setVisible(False)
        body_layout = QtWidgets.QVBoxLayout(self._body)
        body_layout.setContentsMargins(8, 2, 8, 6)
        body_layout.setSpacing(2)
        self._body_layout = body_layout
        outer.addWidget(self._body)

        self._update_header()

    def add_action(self, description: str):
        """Append a new tool-call entry."""
        self._entries.append(description)

        label = QtWidgets.QLabel(f"• {description}")
        label.setWordWrap(True)
        label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        label.setStyleSheet(
            "QLabel { color: #8fbf8f; font-size: 11px; font-family: monospace; border: none; }"
        )
        self._body_layout.addWidget(label)
        self._update_header()

    def add_result(self, output: str):
        """Show a tool's output beneath its entry."""
        if len(output) > _MAX_ACTION_OUTPUT_CHARS:
            output = output[:_MAX_ACTION_OUTPUT_CHARS] + "…"
        label = QtWidgets.QLabel(output)
        label.setTextFormat(Qt.TextFormat.PlainText)
        label.setWordWrap(True)
        label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse |
            Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        label.setStyleSheet(
            "QLabel { color: #9a9a9a; font-size: 10px; font-family: monospace;"
            " border: none; padding-left: 10px; }"
        )
        self._body_layout.addWidget(label)

    def _update_header(self):
        n = len(self._entries)
        expanded = self._body.isVisible()
        arrow = "▼" if expanded else "▶"
        noun = "action" if n == 1 else "actions"
        self._toggle_btn.setText(f"{arrow}  {n} {noun} taken")

    def _toggle(self):
        visible = not self._body.isVisible()
        self._body.setVisible(visible)
        self._update_header()
        # Let the scroll area reflow
        self.updateGeometry()
        if parent := self.parent():
            parent.updateGeometry()


# ---------------------------------------------------------------------------
# Collapsible reasoning widget
# ---------------------------------------------------------------------------

class _CollapsibleReasoning(QtWidgets.QFrame):
    """
    A collapsible widget showing the model's intermediate reasoning summaries.
    Styled in a muted purple to distinguish from green action logs.
    Collapsed by default.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._text = ""
        self._label = None

        self.setStyleSheet(
            "QFrame { background-color: #251f2e; border-radius: 6px; border: 1px solid #5a3f7a; }"
        )

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._toggle_btn = QtWidgets.QPushButton("▶  Thinking...")
        self._toggle_btn.setFlat(True)
        self._toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle_btn.setStyleSheet(
            "QPushButton {"
            "  color: #a07cc5;"
            "  font-size: 11px;"
            "  text-align: left;"
            "  padding: 5px 8px;"
            "  border: none;"
            "  background: transparent;"
            "}"
            "QPushButton:hover { color: #c4a0e8; }"
        )
        self._toggle_btn.clicked.connect(self._toggle)
        outer.addWidget(self._toggle_btn)

        self._body = QtWidgets.QWidget()
        self._body.setVisible(False)
        body_layout = QtWidgets.QVBoxLayout(self._body)
        body_layout.setContentsMargins(8, 2, 8, 6)
        body_layout.setSpacing(4)
        self._body_layout = body_layout
        outer.addWidget(self._body)

    def append_text(self, text: str):
        if self._label is None:
            self._label = QtWidgets.QLabel()
            self._label.setTextFormat(Qt.TextFormat.PlainText)
            self._label.setWordWrap(True)
            self._label.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Ignored,
                QtWidgets.QSizePolicy.Policy.Preferred,
            )
            self._label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse |
                Qt.TextInteractionFlag.TextSelectableByKeyboard
            )
            self._label.setStyleSheet(
                "QLabel { color: #b89fd4; font-size: 11px; font-style: italic; border: none; }"
            )
            self._body_layout.addWidget(self._label)
        self._text += text
        self._label.setText(self._text.strip())

    def mark_done(self):
        """Update header once the agent has finished thinking."""
        arrow = "▼" if self._body.isVisible() else "▶"
        self._toggle_btn.setText(f"{arrow}  Thought")

    def _toggle(self):
        visible = not self._body.isVisible()
        self._body.setVisible(visible)
        # Sync arrow — preserve "Thinking..." vs "Thought" label
        current = self._toggle_btn.text()
        if "▶" in current:
            self._toggle_btn.setText(current.replace("▶", "▼"))
        else:
            self._toggle_btn.setText(current.replace("▼", "▶"))
        self.updateGeometry()
        if parent := self.parent():
            parent.updateGeometry()


# ---------------------------------------------------------------------------
# Chat bubble widget
# ---------------------------------------------------------------------------

class _ChatBubble(QtWidgets.QFrame):
    """A single chat message rendered as a styled label."""

    def __init__(self, text: str, role: str, parent=None):
        """
        Args:
            text: Message content.
            role: "user" or "assistant" or "system"
        """
        super().__init__(parent)
        self.setMinimumWidth(0)

        self._text = text
        self._label = label = QtWidgets.QLabel()
        label.setTextFormat(Qt.TextFormat.PlainText)
        label.setWordWrap(True)
        label.setMinimumWidth(0)
        label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse |
            Qt.TextInteractionFlag.TextSelectableByKeyboard |
            Qt.TextInteractionFlag.LinksAccessibleByMouse
        )
        label.setOpenExternalLinks(True)
        if role == "assistant" and text:
            self.show_final_text(text)
        else:
            label.setText(text)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.addWidget(label)

        if role == "user":
            self.setStyleSheet(
                "QFrame { background-color: #2a5caa; border-radius: 8px; }"
                "QLabel { color: white; }"
            )
        elif role == "assistant":
            self.setStyleSheet(
                "QFrame { background-color: #3a3a3a; border-radius: 8px; }"
                "QLabel { color: #e0e0e0; }"
            )
        else:  # system / info
            self.setStyleSheet(
                "QFrame { background-color: #444; border-radius: 4px; }"
                "QLabel { color: #aaa; font-style: italic; font-size: 11px; }"
            )

    def append_text(self, text: str):
        """Add streamed text, shown plain until the message is complete."""
        self._text += text
        self._label.setText(self._text)

    def show_final_text(self, text: str):
        self._text = text
        self._label.setTextFormat(Qt.TextFormat.MarkdownText)
        self._label.setText(text)


# ---------------------------------------------------------------------------
# Main panel widget
# ---------------------------------------------------------------------------

class _AIPanelWidget(QtWidgets.QWidget):
    """Inner widget for the AI chat panel."""

    def __init__(self, cmd, parent=None, load_file=None):
        super().__init__(parent)
        self._cmd = cmd
        self._load_file = load_file or cmd.load
        self._worker = None
        self._agent = None
        self._settings_dialog = None
        self._model_available = True
        self._current_actions: "_CollapsibleActions | None" = None
        self._current_reasoning: "_CollapsibleReasoning | None" = None
        self._last_reasoning: "_CollapsibleReasoning | None" = None
        self._stream_bubble: "_ChatBubble | None" = None
        self._cancel_requested = False
        self._undo_snapshot = None
        self._turn_tokens = [0, 0]
        self._session_tokens = [0, 0]
        self._sent_messages = []
        self._recall_index = None
        self._init_agent()
        self._build_ui()
        self.setAcceptDrops(True)

    def _init_agent(self):
        try:
            from pmg_qt.ai_agent import AIAgent
            self._agent = AIAgent(self._cmd)
        except Exception as exc:  # noqa: BLE001
            self._agent = None
            self._init_error = str(exc)
        else:
            self._init_error = None
            self._restore_settings()

    def _restore_settings(self):
        from pmg_qt.ai_agent import PROVIDERS, REASONING_EFFORTS

        for provider in PROVIDERS:
            api_key = _load_saved_api_key(provider)
            if api_key:
                self._agent.set_api_key(api_key, provider)

        settings = _settings()
        provider = str(settings.value("provider", ""))
        if provider in PROVIDERS:
            self._agent.set_provider(provider)
        model = str(settings.value(f"model/{self._agent.get_provider()}", ""))
        if model:
            self._agent.set_model(model)
        effort = str(settings.value("reasoning", ""))
        if effort in REASONING_EFFORTS:
            self._agent.set_reasoning_effort(effort)

    def save_settings(self):
        """Remember the provider, its model and the reasoning effort."""
        settings = _settings()
        provider = self._agent.get_provider()
        settings.setValue("provider", provider)
        settings.setValue(f"model/{provider}", self._agent.get_model())
        settings.setValue("reasoning", self._agent.get_reasoning_effort())

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # --- Chat history area ---
        self._scroll_area = QtWidgets.QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._history_widget = QtWidgets.QWidget()
        self._history_layout = QtWidgets.QVBoxLayout(self._history_widget)
        self._history_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._history_layout.setSpacing(6)
        self._history_layout.setContentsMargins(4, 4, 4, 4)

        self._scroll_area.setWidget(self._history_widget)
        layout.addWidget(self._scroll_area, stretch=1)

        # --- Status label ---
        self._status_label = QtWidgets.QLabel("")
        self._status_label.setStyleSheet("color: #aaa; font-style: italic; font-size: 11px;")
        layout.addWidget(self._status_label)

        # --- Preset prompts ---
        preset_layout = QtWidgets.QHBoxLayout()
        preset_layout.setSpacing(4)
        self._preset_btns = []
        for label, prompt in _PRESET_PROMPTS:
            button = QtWidgets.QPushButton(label)
            button.setToolTip(prompt)
            button.clicked.connect(lambda _checked=False, text=prompt: self._send_text(text))
            preset_layout.addWidget(button)
            self._preset_btns.append(button)
        layout.addLayout(preset_layout)

        # --- Session tools and token counts ---
        tools_layout = QtWidgets.QHBoxLayout()
        tools_layout.setSpacing(4)

        self._undo_btn = QtWidgets.QPushButton("Undo AI turn")
        self._undo_btn.setToolTip(
            "Restore the scene to how it was before the AI's last turn. "
            "Manual changes made since then are also undone."
        )
        self._undo_btn.setEnabled(False)
        self._undo_btn.clicked.connect(self._on_undo)
        tools_layout.addWidget(self._undo_btn)

        self._export_btn = QtWidgets.QPushButton("Export .pml")
        self._export_btn.setToolTip("Save the commands the AI has run as a PyMOL script")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._on_export)
        tools_layout.addWidget(self._export_btn)

        tools_layout.addStretch()
        self._usage_label = QtWidgets.QLabel("")
        self._usage_label.setStyleSheet("color: #888; font-size: 10px;")
        tools_layout.addWidget(self._usage_label)
        layout.addLayout(tools_layout)

        # --- Input area ---
        input_layout = QtWidgets.QHBoxLayout()

        self._input = QtWidgets.QTextEdit()
        self._input.setFixedHeight(70)
        self._input.setPlaceholderText(
            "Ask the AI anything about your structure... (Enter to send, Shift+Enter for a new line)"
        )
        self._input.setAcceptDrops(False)  # dropped files load instead of pasting a path
        self._input.installEventFilter(self)
        input_layout.addWidget(self._input, stretch=1)

        btn_layout = QtWidgets.QVBoxLayout()

        self._send_btn = QtWidgets.QPushButton("Send")
        self._send_btn.setFixedWidth(60)
        self._send_btn.clicked.connect(self._on_send)

        self._cancel_btn = QtWidgets.QPushButton("Cancel")
        self._cancel_btn.setFixedWidth(60)
        self._cancel_btn.setStyleSheet("QPushButton { color: #ff6b6b; }")
        self._cancel_btn.clicked.connect(self._on_cancel)
        self._cancel_btn.setVisible(False)

        self._clear_btn = QtWidgets.QPushButton("Clear")
        self._clear_btn.setFixedWidth(60)
        self._clear_btn.clicked.connect(self._on_clear)

        btn_layout.addWidget(self._send_btn)
        btn_layout.addWidget(self._cancel_btn)
        btn_layout.addWidget(self._clear_btn)
        btn_layout.addStretch()

        input_layout.addLayout(btn_layout)
        layout.addLayout(input_layout)

        # Show init error if agent failed to load
        if self._init_error:
            self._add_bubble(
                f"Error initializing AI agent: {self._init_error}\n\n"
                "Make sure 'openai' is installed (pip install openai).",
                "system",
            )
            self._input.setEnabled(False)
            self._send_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # Event filter: Enter submits, Up/Down recall earlier messages
    # ------------------------------------------------------------------

    def eventFilter(self, obj, event):
        if obj is self._input and event.type() == QtCore.QEvent.Type.KeyPress:
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                if not event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    self._on_send()
                    return True
            elif event.key() == Qt.Key.Key_Up and self._recall_previous():
                return True
            elif event.key() == Qt.Key.Key_Down and self._recall_next():
                return True
        return super().eventFilter(obj, event)

    def _is_browsing_sent_messages(self) -> bool:
        text = self._input.toPlainText()
        if self._recall_index is None:
            return not text
        if text == self._sent_messages[self._recall_index]:
            return True
        self._recall_index = None
        return False

    def _show_recalled(self, index):
        self._recall_index = index
        self._input.setPlainText("" if index is None else self._sent_messages[index])
        self._input.moveCursor(QtGui.QTextCursor.MoveOperation.End)

    def _recall_previous(self) -> bool:
        if not self._sent_messages or not self._is_browsing_sent_messages():
            return False
        index = len(self._sent_messages) if self._recall_index is None else self._recall_index
        self._show_recalled(max(index - 1, 0))
        return True

    def _recall_next(self) -> bool:
        if self._recall_index is None or not self._is_browsing_sent_messages():
            return False
        index = self._recall_index + 1
        self._show_recalled(index if index < len(self._sent_messages) else None)
        return True

    # ------------------------------------------------------------------
    # Drag and drop: load structure files
    # ------------------------------------------------------------------

    @staticmethod
    def _dropped_paths(event):
        return [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]

    def dragEnterEvent(self, event):
        if self._worker is None and self._dropped_paths(event):
            event.acceptProposedAction()

    def dropEvent(self, event):
        if self._worker is not None:
            return
        for path in self._dropped_paths(event):
            try:
                self._load_file(path)
            except Exception as exc:  # noqa: BLE001
                self._add_bubble(f"Could not load {path}: {exc}", "system")
            else:
                self._add_bubble(f"Loaded {path}", "system")
        event.acceptProposedAction()
        self._input.setFocus()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_send(self):
        text = self._input.toPlainText().strip()
        if self._send_text(text):
            self._input.clear()
            self._recall_index = None

    def _send_text(self, text: str) -> bool:
        """Start an agent turn. Returns True if the message was accepted."""
        if not text or self._worker is not None or self._agent is None:
            return False
        if not self._model_available:
            self._status_label.setText("Choose an available AI model from AI > AI Settings...")
            return False

        if not self._sent_messages or self._sent_messages[-1] != text:
            self._sent_messages.append(text)
        self._add_bubble(text, "user")
        self._current_actions = None    # fresh action group for this turn
        self._current_reasoning = None  # fresh reasoning group for this turn
        self._last_reasoning = None
        self._stream_bubble = None
        self._cancel_requested = False
        self._turn_tokens = [0, 0]
        self._set_running(True)
        self._status_label.setText("Thinking...")

        self._worker = _AgentWorker(self._agent, text, self)
        self._worker.finished.connect(self._on_agent_finished)
        self._worker.tool_called.connect(self._on_tool_called)
        self._worker.tool_result.connect(self._on_tool_result)
        self._worker.reasoning.connect(self._on_reasoning)
        self._worker.text_delta.connect(self._on_text_delta)
        self._worker.usage.connect(self._on_usage)
        self._worker.confirm_requested.connect(self._on_confirm_requested)
        self._worker.error.connect(self._on_agent_error)
        self._worker.start()
        return True

    def _on_cancel(self):
        if self._agent:
            self._agent.cancel()
        self._cancel_requested = True
        self._status_label.setText("Cancelling...")
        self._cancel_btn.setEnabled(False)

    def _on_reasoning(self, text: str):
        # Each run of reasoning gets its own widget (interleaved with actions).
        # If actions were shown since the last reasoning, start a fresh reasoning widget.
        if self._current_reasoning is None:
            self._current_reasoning = _CollapsibleReasoning(self._history_widget)
            self._last_reasoning = self._current_reasoning
            self._history_layout.addWidget(self._current_reasoning)
        self._current_reasoning.append_text(text)
        # Reset actions so the next tool call starts a new action group after this reasoning
        self._current_actions = None
        QtCore.QTimer.singleShot(50, self._scroll_to_bottom)

    def _on_text_delta(self, text: str):
        if self._stream_bubble is None:
            self._stream_bubble = _ChatBubble("", "assistant", self._history_widget)
            self._history_layout.addWidget(self._stream_bubble)
        self._stream_bubble.append_text(text)
        self._current_actions = None
        self._current_reasoning = None
        QtCore.QTimer.singleShot(50, self._scroll_to_bottom)

    def _on_usage(self, used_in: int, used_out: int):
        self._turn_tokens[0] += used_in
        self._turn_tokens[1] += used_out
        self._session_tokens[0] += used_in
        self._session_tokens[1] += used_out
        self._usage_label.setText(
            f"Last: {self._turn_tokens[0]:,} in / {self._turn_tokens[1]:,} out   "
            f"Session: {sum(self._session_tokens):,}"
        )
        self._usage_label.setToolTip("Tokens sent to and received from the model")

    def _on_confirm_requested(self, command: str, reason: str):
        worker = self.sender()
        answer = QtWidgets.QMessageBox.question(
            self,
            "Allow this AI command?",
            f"The AI wants to run:\n\n{command}\n\nThis needs approval because {reason}.",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        worker.answer_confirmation(answer == QtWidgets.QMessageBox.StandardButton.Yes)

    def _finish_stream_bubble(self, final_text: str = ""):
        """Render a completed streamed message as Markdown."""
        if self._stream_bubble is not None:
            self._stream_bubble.show_final_text(final_text or self._stream_bubble._text)
            self._stream_bubble = None

    def _remember_undo_point(self):
        worker = self._worker
        if worker is not None and self._agent.last_turn_ran_commands and worker.snapshot:
            self._undo_snapshot = worker.snapshot
        self._undo_btn.setEnabled(self._undo_snapshot is not None)
        self._export_btn.setEnabled(self._agent.has_commands())

    def _on_agent_finished(self, response: str):
        self._remember_undo_point()
        self._worker = None
        if self._last_reasoning is not None:
            self._last_reasoning.mark_done()
        self._current_actions = None
        self._current_reasoning = None
        self._last_reasoning = None
        self._status_label.setText("")
        self._set_running(False)
        if self._stream_bubble is not None and response:
            self._finish_stream_bubble(response)
        else:
            self._finish_stream_bubble()
            if response:
                self._add_bubble(response, "assistant")
            elif self._cancel_requested:
                self._add_bubble("Cancelled.", "system")
            else:
                self._add_bubble("The model returned no text.", "system")
        self._input.setFocus()

    def _on_tool_result(self, output: str):
        if self._current_actions is not None:
            self._current_actions.add_result(output)

    def _on_tool_called(self, description: str):
        self._status_label.setText(description)
        self._finish_stream_bubble()
        # Lazily create the collapsible action group. Reset reasoning so the next
        # reasoning chunk starts a fresh widget after this group of actions.
        if self._current_actions is None:
            self._current_actions = _CollapsibleActions(self._history_widget)
            self._history_layout.addWidget(self._current_actions)
        self._current_actions.add_action(description)
        # Reset reasoning so next reasoning chunk appears after these actions
        self._current_reasoning = None
        QtCore.QTimer.singleShot(50, self._scroll_to_bottom)

    def _on_agent_error(self, error_msg: str):
        self._remember_undo_point()
        self._worker = None
        self._finish_stream_bubble()
        self._current_actions = None
        self._current_reasoning = None
        self._last_reasoning = None
        self._status_label.setText("")
        self._set_running(False)
        self._add_bubble(f"Error: {error_msg}", "system")

    def _on_clear(self):
        self._current_actions = None
        self._current_reasoning = None
        self._last_reasoning = None
        self._stream_bubble = None
        self._undo_snapshot = None
        self._undo_btn.setEnabled(False)
        self._export_btn.setEnabled(False)
        if self._agent:
            self._agent.clear_history()
            self._agent.clear_command_log()
        while self._history_layout.count():
            item = self._history_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _on_undo(self):
        if self._undo_snapshot is None or self._worker is not None:
            return
        try:
            self._cmd.set_session(self._undo_snapshot)
        except Exception as exc:  # noqa: BLE001
            self._add_bubble(f"Could not undo: {exc}", "system")
            return
        self._undo_snapshot = None
        self._undo_btn.setEnabled(False)
        self._agent.revert_last_turn()
        self._export_btn.setEnabled(self._agent.has_commands())
        self._add_bubble("Restored the scene to before the AI's last turn.", "system")

    def _on_export(self):
        path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export AI commands", "ai_session.pml", "PyMOL script (*.pml)"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(self._agent.export_script())
        except OSError as exc:
            self._add_bubble(f"Could not save {path}: {exc}", "system")
            return
        self._add_bubble(f"Saved commands to {path}", "system")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _add_bubble(self, text: str, role: str):
        bubble = _ChatBubble(text, role, self._history_widget)
        self._history_layout.addWidget(bubble)
        QtCore.QTimer.singleShot(50, self._scroll_to_bottom)

    def _scroll_to_bottom(self):
        vbar = self._scroll_area.verticalScrollBar()
        vbar.setValue(vbar.maximum())

    def _set_running(self, running: bool):
        """Toggle between idle state (Send visible) and running state (Cancel visible)."""
        self._input.setEnabled(not running)
        self._send_btn.setVisible(not running)
        self._send_btn.setEnabled(not running and self._model_available)
        self._cancel_btn.setVisible(running)
        self._cancel_btn.setEnabled(running)
        self._clear_btn.setEnabled(not running)
        for button in self._preset_btns:
            button.setEnabled(not running)
        self._undo_btn.setEnabled(not running and self._undo_snapshot is not None)
        self._export_btn.setEnabled(not running and self._agent is not None and self._agent.has_commands())

    def is_running(self) -> bool:
        return self._worker is not None

    def agent(self):
        return self._agent

    def set_model_available(self, available: bool, message: str = ""):
        self._model_available = available
        self._send_btn.setEnabled(available and self._worker is None)
        self._status_label.setText(message)

    def open_settings(self):
        if self._settings_dialog is None:
            self._settings_dialog = _AISettingsDialog(self, self)
        self._settings_dialog.show()
        self._settings_dialog.raise_()
        self._settings_dialog.activateWindow()


class _AISettingsDialog(QtWidgets.QDialog):
    """Dialog for configuring the AI provider, model, key, and reasoning mode."""

    def __init__(self, panel: _AIPanelWidget, parent=None):
        super().__init__(parent)
        self._panel = panel
        self._agent = panel.agent()
        self._model_worker = None
        self._model_worker_provider = None
        self.setWindowTitle("AI Settings")
        self.setMinimumWidth(420)
        self._build_ui()
        self._load_current_settings()

    def _build_ui(self):
        from pmg_qt.ai_agent import PROVIDER_LABELS, PROVIDERS, REASONING_EFFORTS

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(8)

        self._provider_combo = QtWidgets.QComboBox()
        for provider in PROVIDERS:
            self._provider_combo.addItem(PROVIDER_LABELS[provider], provider)
        self._provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        form.addRow("Provider:", self._provider_combo)

        model_row = QtWidgets.QHBoxLayout()
        model_row.setSpacing(6)
        self._model_combo = QtWidgets.QComboBox()
        self._model_combo.setMinimumWidth(250)
        self._model_combo.currentIndexChanged.connect(self._on_model_changed)
        model_row.addWidget(self._model_combo, stretch=1)

        self._refresh_models_btn = QtWidgets.QPushButton("Refresh")
        self._refresh_models_btn.clicked.connect(self._refresh_models)
        model_row.addWidget(self._refresh_models_btn)
        form.addRow("Model:", model_row)

        self._api_key_edit = QtWidgets.QLineEdit()
        self._api_key_edit.setPlaceholderText("Paste your API key")
        self._api_key_edit.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self._api_key_edit.editingFinished.connect(self._on_api_key_changed)
        form.addRow("API key:", self._api_key_edit)

        self._reasoning_combo = QtWidgets.QComboBox()
        self._reasoning_combo.addItems(REASONING_EFFORTS)
        self._reasoning_combo.currentTextChanged.connect(self._on_reasoning_mode_changed)
        form.addRow("Reasoning:", self._reasoning_combo)

        layout.addLayout(form)

        self._status_label = QtWidgets.QLabel("")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("color: #aaa; font-style: italic; font-size: 11px;")
        layout.addWidget(self._status_label)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.hide)
        layout.addWidget(buttons)

    def _load_current_settings(self):
        if not self._agent:
            self._set_controls_enabled(False)
            self._status_label.setText("AI agent is not available.")
            return

        self._api_key_edit.setText(self._agent.get_api_key())
        self._reasoning_combo.setCurrentText(self._agent.get_reasoning_effort())
        self._provider_combo.blockSignals(True)
        self._provider_combo.setCurrentIndex(self._provider_combo.findData(self._agent.get_provider()))
        self._provider_combo.blockSignals(False)
        self._model_combo.addItem(self._agent.get_model(), self._agent.get_model())
        self._sync_controls()

    def _set_controls_enabled(self, enabled: bool):
        self._provider_combo.setEnabled(enabled)
        self._model_combo.setEnabled(enabled)
        self._refresh_models_btn.setEnabled(enabled)
        self._api_key_edit.setEnabled(enabled)
        self._reasoning_combo.setEnabled(enabled)

    def _sync_controls(self):
        running = self._panel.is_running()
        provider = self._agent.get_provider() if self._agent else "openai"
        can_edit = not running and self._agent is not None
        self._provider_combo.setEnabled(can_edit)
        self._model_combo.setEnabled(can_edit)
        self._refresh_models_btn.setEnabled(can_edit and self._model_worker is None)
        self._api_key_edit.setEnabled(can_edit and provider != "ollama")
        self._reasoning_combo.setEnabled(can_edit and provider != "ollama")
        if running:
            self._status_label.setText("AI settings are locked while the assistant is running.")

    def showEvent(self, event):
        super().showEvent(event)
        self._sync_controls()

    def _on_api_key_changed(self):
        if not self._agent or self._agent.get_provider() == "ollama":
            return
        api_key = self._api_key_edit.text().strip()
        if api_key == self._agent.get_api_key():
            return
        self._agent.set_api_key(api_key)
        if _save_api_key(self._agent.get_provider(), api_key):
            self._status_label.setText("API key saved to the system keychain.")
        else:
            self._status_label.setText(
                "API key will be used until PyMOL closes. Run 'pip install keyring' to remember it."
            )

    def _on_provider_changed(self, index: int):
        if not self._agent or self._panel.is_running() or index < 0:
            return
        self._on_api_key_changed()
        self._agent.set_provider(self._provider_combo.itemData(index))
        self._api_key_edit.setText(self._agent.get_api_key())
        self._panel.save_settings()
        self._sync_controls()
        self._refresh_models()

    def _on_model_changed(self, index: int):
        if not self._agent or index < 0:
            return
        model_id = self._model_combo.itemData(index)
        if model_id:
            self._agent.set_model(model_id)
            self._panel.set_model_available(True)
            self._panel.save_settings()

    def _on_reasoning_mode_changed(self, mode: str):
        if self._agent:
            self._agent.set_reasoning_effort(mode)
            self._panel.save_settings()

    def _refresh_models(self):
        if not self._agent or self._model_worker is not None or self._panel.is_running():
            return
        self._on_api_key_changed()
        self._status_label.setText("Loading models...")
        self._refresh_models_btn.setEnabled(False)
        self._model_worker_provider = self._agent.get_provider()
        self._model_worker = _ModelDiscoveryWorker(self._agent, self)
        self._model_worker.finished.connect(self._on_models_loaded)
        self._model_worker.error.connect(self._on_models_error)
        self._model_worker.start()

    def _on_models_loaded(self, options):
        worker_provider = self._model_worker_provider
        self._model_worker = None
        self._model_worker_provider = None
        provider = self._agent.get_provider() if self._agent else "openai"
        if worker_provider and worker_provider != provider:
            self._sync_controls()
            return

        from pmg_qt.ai_agent import PROVIDER_LABELS

        self._model_combo.blockSignals(True)
        self._model_combo.clear()
        for option in options:
            self._model_combo.addItem(option.label, option.id)

        if not options:
            if provider == "ollama":
                self._model_combo.addItem("No local vision+tools models", "")
                self._panel.set_model_available(
                    False,
                    "No installed Ollama models advertise both vision and tools.",
                )
                self._status_label.setText("No installed Ollama models advertise both vision and tools.")
            else:
                model_id = self._agent.get_model()
                self._model_combo.addItem(model_id, model_id)
                self._panel.set_model_available(True)
                self._status_label.setText(
                    f"No {PROVIDER_LABELS[provider]} models returned; keeping the current model."
                )
        else:
            current = self._agent.get_model()
            index = self._model_combo.findData(current)
            if index < 0:
                index = 0
            self._model_combo.setCurrentIndex(index)
            model_id = self._model_combo.itemData(index)
            if model_id:
                self._agent.set_model(model_id)
                self._panel.save_settings()
            self._panel.set_model_available(True)
            self._status_label.setText("")

        self._model_combo.blockSignals(False)
        self._sync_controls()

    def _on_models_error(self, error_msg: str):
        worker_provider = self._model_worker_provider
        self._model_worker = None
        self._model_worker_provider = None
        provider = self._agent.get_provider() if self._agent else "openai"
        if worker_provider and worker_provider != provider:
            self._sync_controls()
            return

        self._model_combo.blockSignals(True)
        self._model_combo.clear()
        if self._agent and provider != "ollama":
            model_id = self._agent.get_model()
            self._model_combo.addItem(model_id, model_id)
            self._panel.set_model_available(True)
        else:
            self._model_combo.addItem("No local vision+tools models", "")
            self._panel.set_model_available(False, f"Model refresh failed: {error_msg}")
        self._model_combo.blockSignals(False)
        self._status_label.setText(f"Model refresh failed: {error_msg}")
        self._sync_controls()


# ---------------------------------------------------------------------------
# Public: QDockWidget wrapper (matches BuilderPanelDocked pattern)
# ---------------------------------------------------------------------------

class AIPanel(QtWidgets.QDockWidget):
    """Dockable AI chat panel for PyMOL."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("AI Assistant")
        self.setMinimumWidth(280)
        self.setMaximumWidth(500)

        self._inner = _AIPanelWidget(parent.cmd, self, load_file=parent.load_dialog)
        self.setWidget(self._inner)

    def open_settings(self):
        self._inner.open_settings()
