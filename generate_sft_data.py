import json
import random
from datetime import date, timedelta

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

LANGUAGES = ["English", "Spanish", "French", "German"]
PLANTS = ["VLC01", "MAD01", "BER01", "PAR01"]
UNITS = ["kg"]


def make_scenario():
    unit = random.choice(UNITS)

    return {
        "material_id": f"MAT-{random.randint(1000, 9999)}",
        "quantity": random.randint(10, 5000),
        "unit": unit,
        "required_date": (
            date.today() + timedelta(days=random.randint(2, 90))
        ).isoformat(),
        "plant_id": random.choice(PLANTS),
        "language": random.choice(LANGUAGES),
    }


def generate_user_request(scenario):
    prompt = f"""
Create ONE realistic supply-chain user request in {scenario["language"]}.

The request must express exactly this information:

material_id: {scenario["material_id"]}
quantity: {scenario["quantity"]}
unit: {scenario["unit"]}
required_date: {scenario["required_date"]}
plant_id: {scenario["plant_id"]}

The user wants to know whether this material quantity can be fulfilled
at that plant by the required date.

Requirements:
- Sound like a natural employee request, not synthetic training data.
- Do not explicitly name the API or function.
- You may express dates and quantities naturally.
- Do not change, add, or remove any information.
- Return only the user request.
"""

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=prompt,
        temperature=1.0,
    )

    return response.output_text.strip()


def build_example():
    scenario = make_scenario()
    user_request = generate_user_request(scenario)

    tool_call = {
        "name": "can_fulfill_material_request",
        "arguments": {
            "material_id": scenario["material_id"],
            "quantity": scenario["quantity"],
            "unit": scenario["unit"],
            "required_date": scenario["required_date"],
            "plant_id": scenario["plant_id"],
        },
    }

    return {
        "language": scenario["language"],
        "messages": [
            {
                "role": "user",
                "content": user_request,
            },
            {
                "role": "assistant",
                "tool_call": tool_call,
            },
        ],
    }


if __name__ == "__main__":
    examples = []

    for i in range(3):
        example = build_example()
        examples.append(example)

        print(i, example["messages"][0]["content"])
        print(i, example["messages"][1]["tool_call"])

    with open("sft_data.jsonl", "w") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")