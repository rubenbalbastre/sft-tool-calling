import unittest

from src.environment.env import SupplyChainEnvironment


BASE = {
    "material_id": "MAT-1842",
    "quantity": 350,
    "unit": "kg",
    "required_date": "2026-10-15",
    "city": None,
    "explicit_plant_id": None,
    "selected_plant_id": None,
}


def action(name, **arguments):
    return {"name": name, "arguments": arguments}


class SupplyChainEnvironmentTest(unittest.TestCase):
    def test_explicit_id_goes_directly_to_fulfillment(self):
        scenario = BASE | {"explicit_plant_id": "ES-03"}
        env = SupplyChainEnvironment(scenario, "Can ES-03 fulfill the request?")
        env.reset()
        _, reward, done, info = env.step(action(
            "can_fulfill_material_request",
            material_id="MAT-1842", quantity=350, unit="kg",
            required_date="2026-10-15", plant_id="ES-03",
        ))
        self.assertEqual((reward, done, info["success"]), (1.0, True, True))

    def test_unique_city_lookup_then_fulfillment(self):
        scenario = BASE | {"city": "Bilbao", "selected_plant_id": "ES-09"}
        env = SupplyChainEnvironment(scenario, "Can Bilbao fulfill the request?")
        env.reset()
        _, reward, done, _ = env.step(action("check_location", city="Bilbao"))
        self.assertEqual((reward, done), (0.2, False))
        _, reward, done, _ = env.step(action(
            "can_fulfill_material_request",
            material_id="MAT-1842", quantity=350, unit="kg",
            required_date="2026-10-15", plant_id="ES-09",
        ))
        self.assertEqual((reward, done), (0.8, True))

    def test_ambiguous_city_clarifies_and_uses_user_selection(self):
        scenario = BASE | {"city": "Valencia", "selected_plant_id": "ES-08"}
        env = SupplyChainEnvironment(scenario, "Can Valencia fulfill the request?")
        env.reset()
        env.step(action("check_location", city="Valencia"))
        observation, reward, done, _ = env.step(action(
            "ask_for_clarification", candidate_plant_ids=["ES-03", "ES-08"]
        ))
        self.assertEqual(observation["content"], "Valencia Distribution Centre")
        self.assertEqual((reward, done), (0.2, False))
        _, reward, done, _ = env.step(action(
            "can_fulfill_material_request",
            material_id="MAT-1842", quantity=350, unit="kg",
            required_date="2026-10-15", plant_id="ES-08",
        ))
        self.assertEqual((reward, done), (0.6, True))

    def test_unknown_city_requests_new_location(self):
        scenario = BASE | {
            "city": "Albor",
            "replacement_city": "Bilbao",
            "selected_plant_id": "ES-09",
        }
        env = SupplyChainEnvironment(scenario, "Can Albor fulfill the request?")
        env.reset()
        env.step(action("check_location", city="Albor"))
        observation, reward, done, _ = env.step(action("request_new_location"))
        self.assertEqual(observation["content"], "Bilbao")
        self.assertEqual((reward, done), (0.2, False))
        env.step(action("check_location", city="Bilbao"))
        _, reward, done, info = env.step(action(
            "can_fulfill_material_request",
            material_id="MAT-1842", quantity=350, unit="kg",
            required_date="2026-10-15", plant_id="ES-09",
        ))
        self.assertAlmostEqual(reward, 0.4)
        self.assertTrue(done)
        self.assertEqual(info["episode_return"], 1.0)

    def test_intermediate_reward_weights_are_configurable(self):
        scenario = BASE | {"city": "Valencia", "selected_plant_id": "ES-03"}
        env = SupplyChainEnvironment(
            scenario,
            "Can Valencia fulfill the request?",
            reward_weights={
                "lookup": 0.3,
                "clarification": 0.4,
                "new_location": 0.0,
            },
        )
        env.reset()
        _, lookup_reward, _, _ = env.step(action("check_location", city="Valencia"))
        _, clarification_reward, _, _ = env.step(action(
            "ask_for_clarification", candidate_plant_ids=["ES-03", "ES-08"]
        ))
        _, final_reward, done, info = env.step(action(
            "can_fulfill_material_request",
            material_id="MAT-1842", quantity=350, unit="kg",
            required_date="2026-10-15", plant_id="ES-03",
        ))
        self.assertAlmostEqual(lookup_reward, 0.3)
        self.assertAlmostEqual(clarification_reward, 0.4)
        self.assertAlmostEqual(final_reward, 0.3)
        self.assertTrue(done)
        self.assertAlmostEqual(info["episode_return"], 1.0)

    def test_failure_claws_back_intermediate_rewards(self):
        scenario = BASE | {"city": "Valencia", "selected_plant_id": "ES-03"}
        env = SupplyChainEnvironment(scenario, "Can Valencia fulfill the request?")
        env.reset()
        env.step(action("check_location", city="Valencia"))
        _, reward, done, info = env.step(action(
            "ask_for_clarification", candidate_plant_ids=["ES-03", "WRONG"]
        ))
        self.assertEqual(reward, -0.2)
        self.assertTrue(done)
        self.assertEqual(info["episode_return"], 0.0)

    def test_wrong_transition_fails(self):
        scenario = BASE | {"city": "Valencia", "selected_plant_id": "ES-03"}
        env = SupplyChainEnvironment(scenario, "Can Valencia fulfill the request?")
        env.reset()
        _, reward, done, info = env.step(action(
            "can_fulfill_material_request",
            material_id="MAT-1842", quantity=350, unit="kg",
            required_date="2026-10-15", plant_id="ES-03",
        ))
        self.assertEqual((reward, done, info["success"]), (0.0, True, False))


if __name__ == "__main__":
    unittest.main()
