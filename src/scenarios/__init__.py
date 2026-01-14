"""Scenarios Module"""
from .scenario_a_personalization import (
    ScenarioAParams,
    generate_instance as generate_scenario_a,
    solve_for_D,
)
from .scenario_b_too_much_data import (
    ScenarioBParams,
    generate_instance as generate_scenario_b,
    calculate_leakage,
    calculate_outcome,
)

__all__ = [
    'ScenarioAParams',
    'generate_scenario_a',
    'solve_for_D',
    'ScenarioBParams',
    'generate_scenario_b',
    'calculate_leakage',
    'calculate_outcome',
]
