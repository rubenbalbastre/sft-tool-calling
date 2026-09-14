import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.evaluation.common import create_run_directory, summarize
from src.evaluation.evaluate_openai import run_episode
from src.evaluation.evaluate_local import (
    parse_transformers_response,
    run_episode as run_local_episode,
)
from src.environment.tools import TOOLS

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


class FakeLocalBackend:
    def __init__(self, calls):
        self.calls = iter(calls)
        self.messages = []

    async def generate(self, messages):
        self.messages.append(messages.copy())
        name, arguments = next(self.calls)
        call = {
            "id": f"call_{len(self.messages)}",
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }
        assistant = {"role": "assistant", "content": "", "tool_calls": [call]}
        usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        return assistant, [call], usage


class SmolLM3TokenizerWithoutResponseTemplate:
    def parse_response(self, generated_ids, tools):
        raise AttributeError(
            "This tokenizer does not have a `response_template` for parsing chat responses!"
        )

    def decode(self, generated_ids, skip_special_tokens):
        return generated_ids


class EvaluatorTest(unittest.TestCase):
    def test_smolllm3_xml_tool_call_parser(self):
        output = (
            '<tool_call>{"name":"check_location",'
            '"arguments":{"city":"Bilbao"}}</tool_call>'
        )
        parsed = parse_transformers_response(
            SmolLM3TokenizerWithoutResponseTemplate(), output
        )

        self.assertEqual(
            parsed["tool_calls"][0]["function"],
            {"name": "check_location", "arguments": {"city": "Bilbao"}},
        )

    def test_run_directories_are_numbered(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self.assertEqual(create_run_directory(root).name, "eval-0001")
            self.assertEqual(create_run_directory(root).name, "eval-0002")

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

    def test_ambiguous_local_rollout(self):
        scenario = {
            "scenario_id": "test_local",
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
        backend = FakeLocalBackend([
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

        result = asyncio.run(run_local_episode(backend, row, "prompt", 4))

        self.assertTrue(result["success"])
        self.assertEqual(result["steps"], 3)
        self.assertEqual(result["usage"]["total_tokens"], 45)
        self.assertEqual(backend.messages[2][-1]["role"], "user")


if __name__ == "__main__":
    unittest.main()
