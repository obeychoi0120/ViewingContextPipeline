from __future__ import annotations

import unittest

from src.scene_context_extraction.graph_core.validator import DERIVED_GRAPH_FIELDS, OBSERVABLE_GRAPH_FIELDS, compact_observation, validate_observation


class ValidatorCompactTests(unittest.TestCase):
    def test_validate_observation_can_be_compacted_to_observable_fields(self) -> None:
        normalized, warnings = validate_observation(
            {
                "scene_type": "graphic",
                "visual_style_cues": {
                    "media_form": "mixed",
                    "fantasy_element": "medium",
                    "shot_scale": "close-up",
                    "graphic_density": "high",
                    "composition_density": "busy",
                },
                "people_density": "single",
                "face_prominence": "medium",
                "mood_bin": "serious_focused",
                "affect_cues": ["serious", "serious"],
                "scene_function": "news",
                "setting": "news studio",
                "entities": [
                    {
                        "local_id": "e1",
                        "category": "person",
                        "name": "Anchor",
                        "role": "main subject",
                        "relations": {"doing": "reporting", "with": "e2", "AT": "studio"},
                    }
                ],
                "content_axes_4d": {"subject_sociality": 1.0},
                "visual_style": {"primary": "editorial_portrait"},
                "style": "editorial_portrait",
                "mood_state": {"primary": "serious"},
                "mood": "serious",
            }
        )
        compact = compact_observation(normalized)

        self.assertEqual(set(compact), set(OBSERVABLE_GRAPH_FIELDS))
        self.assertTrue(DERIVED_GRAPH_FIELDS.isdisjoint(compact))
        self.assertEqual(compact["scene_type"], "graphic_information")
        self.assertEqual(compact["visual_style_cues"]["media_form"], "unknown")
        self.assertEqual(compact["people_density"], "one")
        self.assertEqual(compact["entities"][0]["relations"], {"DOING": "reporting", "INTERACTS_WITH": "e2", "AT": "studio"})
        self.assertTrue(warnings)

    def test_validate_observation_defaults_invalid_values(self) -> None:
        normalized, _ = validate_observation({"entities": "bad"})

        self.assertEqual(normalized["scene_type"], "unknown")
        self.assertEqual(normalized["entities"], [])

    def test_validate_observation_accepts_low_risk_ondevice_aliases(self) -> None:
        normalized, _ = validate_observation(
            {
                "people_density": "two",
                "mood_bin": "excited",
                "affect_cues": ["happy", "focused", "serene", "surprised", "alarmed"],
            }
        )

        self.assertEqual(normalized["people_density"], "few")
        self.assertEqual(normalized["mood_bin"], "cheerful")
        self.assertEqual(
            normalized["affect_cues"],
            ["cheerful", "serious", "calm", "excited", "tense"],
        )


if __name__ == "__main__":
    unittest.main()
