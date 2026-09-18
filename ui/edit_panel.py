"""
TransitionEditPanel — the state of the marker currently being edited, and the
mouse equivalent of every editing keystroke.

Deliberately dumb: it renders a `SelectionInfo` and emits button clicks. It owns
no edits, reads nothing from the graph, and makes no decisions — MainWindow
assembles the info and applies the results, so the panel can be tested by
handing it a value and reading the labels back.

Two things it shows that nothing else can:

- **Both frame numbers at once.** Nudging pushes the other end when the two
  would cross (see core.manual.set_frame), and on an instantaneous transition
  first-light and fully-lit sit on the SAME frame, where the graph can only draw
  one triangle. Without both numbers here, a push would be invisible and a
  coincident pair indistinguishable.
- **What the algorithm said**, beside what the user chose. That is what makes
  Reset a meaningful offer rather than a button whose effect you have to guess.
"""

from dataclasses import dataclass

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

_DIM = "#888888"
_ACTIVE = "#00d2ff"      # matches the graph's selection ring
_EDITED = "#ffffff"      # matches the graph's manual-marker outline


@dataclass(frozen=True)
class SelectionInfo:
    """Everything the panel needs about the current selection."""

    roi: str                       # "original" | "display"
    polarity: str                  # "rising" | "falling"
    which: str                     # "first" | "full" — the active end
    first_frame: int
    full_frame: int
    auto_first: int | None         # None when the transition could not be
    auto_full: int | None          # characterized automatically at all
    deleted: bool
    has_edit: bool                 # this transition carries a manual edit
    position: int                  # 1-based; 0 when it is not a navigation stop
    total: int
    edit_count: int                # edits across the whole clip

    @property
    def first_edited(self) -> bool:
        return self.auto_first is not None and self.auto_first != self.first_frame

    @property
    def full_edited(self) -> bool:
        return self.auto_full is not None and self.auto_full != self.full_frame


def roi_label(roi: str) -> str:
    return "Source" if roi == "original" else "Display"


def polarity_label(polarity: str) -> str:
    return "dark→light" if polarity == "rising" else "light→dark"


class TransitionEditPanel(QGroupBox):

    nudge_requested = pyqtSignal(int)     # -1 / +1 frames
    set_to_playhead_requested = pyqtSignal()
    delete_toggled = pyqtSignal()
    reset_requested = pyqtSignal()
    reset_all_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("Transition Editing", parent)
        outer = QVBoxLayout(self)
        outer.setSpacing(4)

        header = QHBoxLayout()
        self.title_label = QLabel("No transition selected")
        self.count_label = QLabel("")
        self.count_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        header.addWidget(self.title_label)
        header.addStretch()
        header.addWidget(self.count_label)
        outer.addLayout(header)

        frames = QHBoxLayout()
        self.first_label = QLabel("")
        self.full_label = QLabel("")
        frames.addWidget(self.first_label)
        frames.addSpacing(16)
        frames.addWidget(self.full_label)
        frames.addStretch()
        outer.addLayout(frames)

        buttons = QHBoxLayout()
        self.nudge_back_btn = QPushButton("◀")
        self.nudge_fwd_btn = QPushButton("▶")
        self.set_btn = QPushButton("Set to playhead")
        self.delete_btn = QPushButton("Delete")
        self.reset_btn = QPushButton("Reset")
        self.reset_all_btn = QPushButton("Reset All")
        self.nudge_back_btn.setToolTip("Move this marker one frame earlier (Shift+←)")
        self.nudge_fwd_btn.setToolTip("Move this marker one frame later (Shift+→)")
        self.set_btn.setToolTip("Move this marker to the current frame (M)")
        self.delete_btn.setToolTip(
            "Drop this transition from pairing entirely (Delete).\n"
            "Use it on a false read so the real transition beside it can be matched."
        )
        self.reset_btn.setToolTip("Restore this transition to what the algorithm measured")
        self.reset_all_btn.setToolTip("Discard every manual edit on this clip")
        for btn in self._all_buttons():
            # Same rule as every other button here: a focused button swallows
            # the navigation keys before MainWindow.keyPressEvent can see them.
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            buttons.addWidget(btn)
        buttons.addStretch()
        outer.addLayout(buttons)

        self.nudge_back_btn.clicked.connect(lambda: self.nudge_requested.emit(-1))
        self.nudge_fwd_btn.clicked.connect(lambda: self.nudge_requested.emit(1))
        self.set_btn.clicked.connect(self.set_to_playhead_requested)
        self.delete_btn.clicked.connect(self.delete_toggled)
        self.reset_btn.clicked.connect(self.reset_requested)
        self.reset_all_btn.clicked.connect(self.reset_all_requested)

        self.show_selection(None)

    def _all_buttons(self) -> tuple[QPushButton, ...]:
        return (
            self.nudge_back_btn, self.nudge_fwd_btn, self.set_btn,
            self.delete_btn, self.reset_btn, self.reset_all_btn,
        )

    def show_selection(self, info: SelectionInfo | None, edit_count: int = 0) -> None:
        """Render `info`, or the empty state when nothing is selected.

        `edit_count` is used only for the empty state — a selection carries its
        own count — so the "3 edits" readout survives clearing the selection and
        the user never loses sight of having edited something."""
        if info is None:
            self.title_label.setText("No transition selected")
            self.title_label.setStyleSheet(f"color: {_DIM};")
            self.first_label.setText("Click a marker on the graph, or press ↑/↓")
            self.first_label.setStyleSheet(f"color: {_DIM};")
            self.full_label.setText("")
            self.count_label.setText(self._count_text(edit_count))
            for btn in self._all_buttons():
                btn.setEnabled(False)
            self.reset_all_btn.setEnabled(edit_count > 0)
            self.delete_btn.setText("Delete")
            return

        self.title_label.setStyleSheet("")
        title = f"{roi_label(info.roi)} · {polarity_label(info.polarity)}"
        if info.deleted:
            title += "  —  deleted"
        elif info.position:
            title += f"  —  transition {info.position} of {info.total}"
        self.title_label.setText(title)
        self.count_label.setText(self._count_text(info.edit_count))

        self.first_label.setText(
            self._end_text("first-light", info.first_frame, info.auto_first, info.which == "first")
        )
        self.full_label.setText(
            self._end_text("fully-lit", info.full_frame, info.auto_full, info.which == "full")
        )
        self.first_label.setStyleSheet(self._end_style(info.which == "first", info.first_edited))
        self.full_label.setStyleSheet(self._end_style(info.which == "full", info.full_edited))

        for btn in self._all_buttons():
            btn.setEnabled(True)
        # A deleted transition has no marker to move, so only un-deleting and
        # resetting mean anything on one.
        for btn in (self.nudge_back_btn, self.nudge_fwd_btn, self.set_btn):
            btn.setEnabled(not info.deleted)
        self.delete_btn.setText("Restore" if info.deleted else "Delete")
        # Keyed on whether an edit exists at all, not on whether the frames
        # differ: a marker nudged away and back still carries a pinned value
        # that Reset is the way to clear.
        self.reset_btn.setEnabled(info.has_edit)

    @staticmethod
    def _count_text(count: int) -> str:
        if count == 0:
            return ""
        return f"{count} manual edit" + ("" if count == 1 else "s")

    @staticmethod
    def _end_text(name: str, frame: int, auto: int | None, active: bool) -> str:
        marker = "▸ " if active else "   "
        text = f"{marker}{name}  {frame}"
        if auto is not None and auto != frame:
            text += f"  (auto {auto})"
        return text

    @staticmethod
    def _end_style(active: bool, edited: bool) -> str:
        if active:
            return f"color: {_ACTIVE}; font-weight: bold;"
        return f"color: {_EDITED};" if edited else f"color: {_DIM};"
