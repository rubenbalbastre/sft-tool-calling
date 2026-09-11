"""Stateful, deterministically verifiable environment for RLVR rollouts."""

import json

from environment.tools import (
    ask_for_clarification,
    can_fulfill_material_request,
    check_location,
    request_new_location,
)


class SupplyChainEnvironment:
    """Verify tool trajectories without comparing against a gold conversation."""

    def __init__(
        self,
        scenario,
        user_request,
        reward_weights=None,
        success_reward=1.0,
        failure_reward=0.0,
    ):
        self.scenario = scenario
        self.user_request = user_request
        self.reward_weights = {
            "lookup": 0.2,
            "clarification": 0.2,
            **(reward_weights or {}),
        }
        self.success_reward = success_reward
        self.failure_reward = failure_reward
        unknown_weights = set(self.reward_weights) - {"lookup", "clarification"}
        if unknown_weights:
            raise ValueError(f"Unknown intermediate reward weights: {unknown_weights}")
        if any(weight < 0 for weight in self.reward_weights.values()):
            raise ValueError("Intermediate reward weights cannot be negative")
        if sum(self.reward_weights.values()) > success_reward:
            raise ValueError("Intermediate reward weights cannot exceed success_reward")
        self.state = None
        self.matches = []
        self.selected_plant_id = None
        self.history = []
        self.done = False
        self.return_so_far = 0.0

    def reset(self):
        """Start a fresh episode and return the initial user observation."""
        self.matches = []
        self.selected_plant_id = None
        self.history = []
        self.done = False
        self.return_so_far = 0.0
        self.state = (
            "expect_fulfillment"
            if self.scenario.get("explicit_plant_id")
            else "expect_lookup"
        )
        observation = {"role": "user", "content": self.user_request}
        self.history.append(observation)
        return observation

    def step(self, action):
        """Apply one model action and return observation, reward, done, and info."""
        if self.done:
            raise RuntimeError("Episode is finished; call reset() before step()")

        try:
            name, arguments = self._parse_action(action)
            observation = self._transition(name, arguments)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            return self._failure(str(error))

        self.history.append({"role": "assistant", "action": action})
        if observation is not None:
            self.history.append(observation)

        if self.state == "success":
            self.done = True
            reward = self.success_reward - self.return_so_far
            self.return_so_far += reward
            return observation, reward, True, {
                "success": True,
                "episode_return": self.return_so_far,
            }

        reward_name = {
            "check_location": "lookup",
            "ask_for_clarification": "clarification",
        }.get(name)
        reward = self.reward_weights.get(reward_name, 0.0)
        self.return_so_far += reward
        return observation, reward, False, {
            "success": False,
            "state": self.state,
            "episode_return": self.return_so_far,
        }

    def _transition(self, name, arguments):
        if self.state == "expect_lookup":
            return self._lookup(name, arguments)
        if self.state == "expect_clarification":
            return self._clarify(name, arguments)
        if self.state == "expect_new_location_request":
            return self._request_new_location(name, arguments)
        if self.state == "expect_fulfillment":
            return self._fulfill(name, arguments)
        raise ValueError(f"Unknown environment state: {self.state}")

    def _lookup(self, name, arguments):
        self._require_tool(name, "check_location")
        expected = {"city": self.scenario["city"]}
        if arguments != expected:
            raise ValueError(f"Expected lookup arguments {expected}, got {arguments}")

        result = check_location(**arguments)
        self.matches = result["matches"]
        if not self.matches:
            self.state = "expect_new_location_request"
        elif len(self.matches) == 1:
            self.selected_plant_id = self.matches[0]["plant_id"]
            self.state = "expect_fulfillment"
        else:
            self.state = "expect_clarification"
        return self._tool_observation(name, result)

    def _clarify(self, name, arguments):
        self._require_tool(name, "ask_for_clarification")
        expected_ids = [plant["plant_id"] for plant in self.matches]
        if arguments != {"candidate_plant_ids": expected_ids}:
            raise ValueError("Clarification candidates must exactly match lookup results")

        result = ask_for_clarification(**arguments)
        selected_id = self.scenario.get("selected_plant_id")
        selected = next(
            (plant for plant in self.matches if plant["plant_id"] == selected_id),
            None,
        )
        if selected is None:
            raise ValueError("Scenario selection is not one of the lookup matches")

        self.selected_plant_id = selected["plant_id"]
        self.state = "expect_fulfillment"
        return {
            "role": "user",
            "content": selected["name"],
            "tool_result": result,
        }

    def _request_new_location(self, name, arguments):
        self._require_tool(name, "request_new_location")
        if arguments:
            raise ValueError("request_new_location takes no arguments")
        result = request_new_location()
        self.state = "success"
        return self._tool_observation(name, result)

    def _fulfill(self, name, arguments):
        self._require_tool(name, "can_fulfill_material_request")
        plant_id = self.scenario.get("explicit_plant_id") or self.selected_plant_id
        expected = {
            "material_id": self.scenario["material_id"],
            "quantity": self.scenario["quantity"],
            "unit": self.scenario["unit"],
            "required_date": self.scenario["required_date"],
            "plant_id": plant_id,
        }
        if arguments != expected:
            raise ValueError(f"Expected fulfillment arguments {expected}, got {arguments}")

        result = can_fulfill_material_request(**arguments)
        self.state = "success"
        return self._tool_observation(name, result)

    def _failure(self, reason):
        self.state = "failure"
        self.done = True
        reward = self.failure_reward - self.return_so_far
        self.return_so_far += reward
        return None, reward, True, {
            "success": False,
            "reason": reason,
            "episode_return": self.return_so_far,
        }

    @staticmethod
    def _require_tool(actual, expected):
        if actual != expected:
            raise ValueError(f"Expected {expected}, got {actual}")

    @staticmethod
    def _tool_observation(name, result):
        return {"role": "tool", "name": name, "content": result}

    @staticmethod
    def _parse_action(action):
        """Accept a compact action or an OpenAI-style function tool call."""
        if "function" in action:
            action = action["function"]
        name = action["name"]
        arguments = action.get("arguments", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(arguments, dict):
            raise TypeError("Tool arguments must be a JSON object")
        return name, arguments
