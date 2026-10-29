# File: test_memguard_after_stop.py
# Location: tests/test_memguard_after_stop.py
# Purpose: mem_guard --after-stop: runs once after a breach, never blocks or raises, and its log line is not a guard event.
# Dependencies: flightrec.mem_guard, flightrec.memguard_events

"""No docker, no kill: the memory reading and the victim / container steps are replaced."""

import io
import time

import pytest

from flightrec import mem_guard as g
from flightrec import memguard_events as mg


def _wait_for(path, seconds=5.0):
    """True once the file exists (the command runs detached)."""
    end = time.time() + seconds
    while time.time() < end:
        if path.exists():
            return True
        time.sleep(0.05)
    return False


@pytest.fixture
def breach(monkeypatch):
    """A guard step with memory below the floor and nothing to kill or stop."""
    monkeypatch.setattr(g, "mem_available_gb", lambda: 0.5)
    monkeypatch.setattr(g, "_victims", lambda: [])
    monkeypatch.setattr(g, "_stop_model_containers", lambda log: None)


def test_after_stop_runs_once_after_a_breach_and_is_logged(breach, tmp_path):
    """The command starts after the local stops; the log says it started."""
    marker, log = tmp_path / "ran", io.StringIO()
    assert g.enforce(1.0, log, f"touch {marker}") is True
    assert _wait_for(marker)
    assert "after-stop started: touch" in log.getvalue()


def test_no_command_means_nothing_extra(breach, tmp_path):
    """The default (empty) command changes nothing: no process, no extra log line."""
    log = io.StringIO()
    assert g.enforce(1.0, log) is True
    assert "after-stop" not in log.getvalue()


def test_above_the_floor_the_command_does_not_run(monkeypatch, tmp_path):
    """No breach, no command."""
    monkeypatch.setattr(g, "mem_available_gb", lambda: 50.0)
    marker = tmp_path / "ran"
    assert g.enforce(1.0, io.StringIO(), f"touch {marker}") is False
    assert not _wait_for(marker, 0.5)


def test_a_failing_command_never_raises(breach):
    """A command that exits non-zero is the operator's problem, not the guard's."""
    assert g.enforce(1.0, io.StringIO(), "exit 3") is True


def test_the_log_line_is_not_a_guard_event():
    """memguard_events keeps only warn / breach / kill / docker stop lines; the new line is skipped."""
    assert mg.classify("  after-stop started: touch /tmp/x") is None
    assert mg.classify("  after-stop failed to start: boom") is None
    assert mg.classify("[03:04:09] BREACH avail=0.5GB < 1.0GB — killing runaways") == "breach"
