import json
import unittest
from types import SimpleNamespace

from evaluation.evaluate_openai import run_episode, summarize
from environment.tools import TOOLS

class FakeResponses:
    def __init__(self, calls):
        self.calls = iter(calls)
        self.index = 0
        self.requests = []

    def create(self, **request):
        self.requests.append(request)
        self.index += 1
        name, arguments = next(self.calls)
        call = SimpleNamespace(
            type="function_call",
            name=name,
            arguments=json.dumps(arguments),
            call_id=f"call_{self.index}",
        )
        usage = SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15)
        return SimpleNamespace(output=[call], usage=usage, id=f"response_{self.index}")


class EvaluatorTest(unittest.TestCase):
    def test_ambiguous_online_rollout(self):
        scenario = {
            "scenario_id": "test_1",
            "kind": "ambiguous",
            "language": "English",
            "difficulty": "simple",
            "material_id": "MAT-1842",
            "quantity": 350,
            "unit": "kg",
            "required_date": "2026-10-15",
            "city": "Valencia",
            "explicit_plant_id": None,
            "selected_plant_id": "ES-08",
        }
        row = {
            "scenario": scenario,
            "messages": [{"role": "user", "content": "Can Valencia supply it?"}],
        }
        responses = FakeResponses([
            ("check_location", {"city": "Valencia"}),
            ("ask_for_clarification", {
                "candidate_plant_ids": ["ES-03", "ES-08"]
            }),
            ("can_fulfill_material_request", {
                "material_id": "MAT-1842",
                "quantity": 350,
                "unit": "kg",
                "required_date": "2026-10-15",
                "plant_id": "ES-08",
            }),
        ])
        client = SimpleNamespace(responses=responses)

        result = run_episode(client, row, "test-model", "prompt", "none", 4)
        self.assertTrue(result["success"])
        self.assertEqual(result["steps"], 3)
        self.assertEqual(result["usage"]["total_tokens"], 45)
        self.assertEqual(summarize([result])["overall"]["success_rate"], 1.0)
        self.assertEqual(len(responses.requests), 3)
        self.assertTrue(all(request["tools"] == TOOLS for request in responses.requests))
        self.assertTrue(all(
            request["reasoning"] == {"effort": "none"}
            for request in responses.requests
        ))


if __name__ == "__main__":
    unittest.main()
