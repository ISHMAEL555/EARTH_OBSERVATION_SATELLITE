"""
Integration Test

Module
------
NadirPointingScenario

Objective
---------
Verify that the NadirPointingScenario correctly
orchestrates the mission execution flow.

This test validates subsystem interfaces only.
"""

import pytest

from scenarios.nadir_pointing.scenario import (
    NadirPointingScenario,
)


# ======================================================
# Dummy Objects
# ======================================================

class DummySimulation:
    pass


class DummySimulator:

    def __init__(self):
        self.finished = False
        self.step_count = 0

    def step(self):
        self.step_count += 1


class DummyGuidance:
    pass


class DummyLogger:
    pass


# ======================================================
# Tests
# ======================================================

def test_scenario_initialization():

    scenario = NadirPointingScenario(
        simulation=DummySimulation(),
        simulator=DummySimulator(),
        guidance=DummyGuidance(),
        logger=DummyLogger(),
    )

    assert scenario.sim is not None
    assert scenario.simulator is not None
    assert scenario.guidance is not None
    assert scenario.logger is not None


def test_initialize(monkeypatch):

    called = {"initialize": False}

    def fake_initialize(self):
        called["initialize"] = True

    monkeypatch.setattr(
        "scenarios.nadir_pointing.scenario.initialize",
        fake_initialize,
    )

    scenario = NadirPointingScenario(
        DummySimulation(),
        DummySimulator(),
        DummyGuidance(),
        DummyLogger(),
    )

    scenario.initialize()

    assert called["initialize"]


def test_update(monkeypatch):

    calls = []

    monkeypatch.setattr(
        "scenarios.nadir_pointing.scenario.propagate",
        lambda self: calls.append("propagate"),
    )

    monkeypatch.setattr(
        "scenarios.nadir_pointing.scenario.update_environment",
        lambda self: calls.append("environment"),
    )

    monkeypatch.setattr(
        "scenarios.nadir_pointing.scenario.update_guidance",
        lambda self: calls.append("guidance"),
    )

    monkeypatch.setattr(
        "scenarios.nadir_pointing.scenario.update_control",
        lambda self: calls.append("control"),
    )

    monkeypatch.setattr(
        "scenarios.nadir_pointing.scenario.log",
        lambda self: calls.append("log"),
    )

    simulator = DummySimulator()

    scenario = NadirPointingScenario(
        DummySimulation(),
        simulator,
        DummyGuidance(),
        DummyLogger(),
    )

    scenario.update()

    assert calls == [
        "propagate",
        "environment",
        "guidance",
        "control",
        "log",
    ]

    assert simulator.step_count == 1


def test_finalize(monkeypatch):

    monkeypatch.setattr(
        "scenarios.nadir_pointing.scenario.finalize",
        lambda self: {"status": "success"},
    )

    scenario = NadirPointingScenario(
        DummySimulation(),
        DummySimulator(),
        DummyGuidance(),
        DummyLogger(),
    )

    result = scenario.finalize()

    assert result["status"] == "success"