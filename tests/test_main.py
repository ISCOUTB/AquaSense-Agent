from __future__ import annotations

import main


def test_main_runs_agent_for_every_scenario_with_separate_outputs(monkeypatch):
    calls: list[tuple[str, list[str]]] = []

    def fake_execute(step: str, arguments: list[str]) -> float:
        calls.append((step, arguments))
        return 0.1

    monkeypatch.setattr(main, "ejecutar", fake_execute)

    assert main.main([
        "--solo", "agent", "--todos-escenarios", "--limite", "12",
    ]) == 0

    assert len(calls) == 10
    assert all(step == "agent" for step, _ in calls)
    for number, (_, arguments) in enumerate(calls, start=1):
        scenario = f"Scenario-{number}"
        assert arguments[arguments.index("--escenario") + 1] == scenario
        assert arguments[arguments.index("--log") + 1] == f"logs/agent_events_{scenario}.csv"
        assert arguments[arguments.index("--resumen-out") + 1] == f"logs/agent_run_{scenario}.json"
        assert arguments[arguments.index("--limite") + 1] == "12"
