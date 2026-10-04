import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.evaluation.common import create_run_directory, summarize
from src.evaluation.evaluate_openai import load_eval_defaults, run_episode
from src.evaluation.evaluate_local import (
    parse_transformers_response,
    run_episode as run_local_episode,
)
from src.evaluation.vllm import quantization_arguments
from src.environment.procurement import ProcurementEnvironment, generate_scenarios
from src.environment.procurement.tools import TOOLS

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
    def parse_response(self, generated_ids, prefix, tools):
        raise AttributeError(
            "This tokenizer does not have a `response_template` for parsing chat responses!"
        )

    def decode(self, generated_ids, skip_special_tokens):
        return generated_ids


class EvaluatorTest(unittest.TestCase):
    def test_openai_evaluator_uses_shared_defaults(self):
        defaults = load_eval_defaults()
        self.assertEqual(defaults["reasoning_effort"], "medium")
        self.assertEqual(defaults["temperature"], 0.0)
        self.assertEqual(defaults["episodes"], 5)

    def test_openai_tool_schemas_are_strict(self):
        for tool in TOOLS:
            parameters = tool["parameters"]
            self.assertEqual(
                set(parameters["required"]), set(parameters["properties"])
            )
            self.assertFalse(parameters["additionalProperties"])

    def test_vllm_quantization_arguments(self):
        self.assertEqual(quantization_arguments("none"), [])
        self.assertEqual(
            quantization_arguments("bnb_4bit"),
            ["--quantization", "bitsandbytes"],
        )

    def test_smolllm3_xml_tool_call_parser(self):
        output = (
            '<tool_call>{"name":"search_suppliers",'
            '"arguments":{"material_id":"MAT-1042"}}</tool_call>'
        )
        parsed = parse_transformers_response(
            SmolLM3TokenizerWithoutResponseTemplate(), output, []
        )

        self.assertEqual(
            parsed["tool_calls"][0]["function"],
            {"name": "search_suppliers", "arguments": {"material_id": "MAT-1042"}},
        )

    def test_run_directories_are_numbered(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self.assertEqual(create_run_directory(root).name, "eval-0001")
            self.assertEqual(create_run_directory(root).name, "eval-0002")

    def test_direct_supplier_online_rollout(self):
        scenario = generate_scenarios(1, "test", 77)[0]
        env = ProcurementEnvironment(scenario)
        quote = env._generate_quote(scenario["requested_supplier_id"])
        best = max(env.oracle_options(), key=lambda option: option["utility"])
        row = {"scenario": scenario}
        responses = FakeResponses([
            ("request_quote", {
                "supplier_id": scenario["requested_supplier_id"],
                "material_id": scenario["material_id"],
                "quantity": scenario["quantity"],
                "unit": scenario["unit"],
                "required_date": scenario["required_date"],
            }),
            ("get_delivery_options", {
                "quote_id": quote["quote_id"],
                "destination": scenario["destination"],
            }),
            ("submit_procurement_plan", {
                "quote_id": best["quote_id"],
                "delivery_option_id": best["delivery_option_id"],
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
        self.assertTrue(all(
            request["temperature"] == 0.0 for request in responses.requests
        ))

    def test_direct_supplier_local_rollout(self):
        scenario = generate_scenarios(1, "test", 88)[0]
        env = ProcurementEnvironment(scenario)
        quote = env._generate_quote(scenario["requested_supplier_id"])
        best = max(env.oracle_options(), key=lambda option: option["utility"])
        row = {"scenario": scenario}
        backend = FakeLocalBackend([
            ("request_quote", {
                "supplier_id": scenario["requested_supplier_id"],
                "material_id": scenario["material_id"],
                "quantity": scenario["quantity"],
                "unit": scenario["unit"],
                "required_date": scenario["required_date"],
            }),
            ("get_delivery_options", {
                "quote_id": quote["quote_id"],
                "destination": scenario["destination"],
            }),
            ("submit_procurement_plan", {
                "quote_id": best["quote_id"],
                "delivery_option_id": best["delivery_option_id"],
            }),
        ])

        result = asyncio.run(run_local_episode(backend, row, "prompt", 4))

        self.assertTrue(result["success"])
        self.assertEqual(result["steps"], 3)
        self.assertEqual(result["usage"]["total_tokens"], 45)
        self.assertEqual(backend.messages[2][-1]["role"], "tool")


if __name__ == "__main__":
    unittest.main()
