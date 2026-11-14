"""Minimal legal baseline for the experiment interface."""


def choose_action(state):
    return state["legal_actions"][0]
