import tempfile
import unittest
from pathlib import Path

from bench import build_hub, human_form


class AlphaTest(unittest.TestCase):
    def test_units_match_two_rater_formula(self):
        pairs = [(1, 1), (2, 2), (1, 2), (3, 3), (2, 1), (4, 4)]
        units = [list(p) for p in pairs]
        self.assertAlmostEqual(human_form.alpha_units(units), human_form.alpha_nominal(pairs))

    def test_perfect_agreement_is_one(self):
        self.assertEqual(human_form.alpha_units([[1, 1, 1], [2, 2, 2], [3, 3]]), 1.0)

    def test_hand_computed_three_raters(self):
        # Units (1,1,2) and (2,2,2), each ordered pair weighted 1/(3-1): o11 = 1, o12 = o21 = 1, o22 = 3;
        # n1 = 2, n2 = 4, n = 6. D_o = 2/6, D_e = 2*2*4/(6*5) = 16/30, alpha = 1 - (1/3)/(8/15) = 0.375.
        self.assertAlmostEqual(human_form.alpha_units([[1, 1, 2], [2, 2, 2]]), 0.375)

    def test_single_value_units_are_skipped(self):
        self.assertEqual(human_form.alpha_units([[1], [1, 1], [2, 2]]), 1.0)


class SummaryTest(unittest.TestCase):
    key = {n: {"n": n, "id": chr(96 + n), "gold": n} for n in (1, 2, 3)}

    def test_majority_and_disputes(self):
        readers = [({1: 1, 2: 2, 3: 1}, {3: "iki seçenek"}, "Evet"),
                   ({1: 1, 2: 2, 3: 1}, {}, "Evet"),
                   ({1: 1, 2: 1, 3: 3}, {}, "Hayır")]
        s = human_form.summarise(self.key, readers)
        self.assertEqual((s["readers"], s["native"], s["majority_agree"], s["no_majority"]), (3, 2, 2, 0))
        self.assertEqual(s["agree_per_reader"], [2, 2, 2])
        self.assertEqual([d["n"] for d in s["disputed"]], [3])
        self.assertEqual(s["disputed"][0]["notes"], ["iki seçenek"])

    def test_tie_is_no_majority(self):
        readers = [({1: 1, 2: 2, 3: 3}, {}, "Evet"), ({1: 2, 2: 2, 3: 3}, {}, "Evet")]
        s = human_form.summarise(self.key, readers)
        self.assertEqual((s["majority_agree"], s["no_majority"]), (2, 1))

    def test_read_answers_and_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "okur_01.md").write_text("# ana dil: Evet\n01: 2 | not\n02: 1\n", encoding="utf-8")
            Path(tmp, "okur_02.md").write_text("01: 2\n", encoding="utf-8")
            paths = human_form.expand([tmp])
            self.assertEqual([p.name for p in paths], ["okur_01.md", "okur_02.md"])
            answers, notes, native = human_form.read_answers(paths[0])
            self.assertEqual((answers, notes, native), ({1: 2, 2: 1}, {1: "not"}, "Evet"))


class CardSentenceTest(unittest.TestCase):
    def test_missing_report_leaves_mark(self):
        self.assertIn("[insert", build_hub.human_sentence(None))

    def test_disputed_items_need_a_note(self):
        human = {"items": 40, "readers": 6, "native": 6, "majority_agree": 39, "no_majority": 0,
                 "mean_agree": 37.5, "min_agree": 35, "max_agree": 40, "alpha_readers": 0.91,
                 "disputed": [{"n": 7}]}
        note = build_hub.REPORTS / "human_disputed_note.txt"
        if note.exists():
            self.skipTest("a real note exists")
        text = build_hub.human_sentence(human)
        self.assertIn("6 native speakers", text)
        self.assertIn("[insert: check the 1 items", text)

    def test_single_reader_sentence(self):
        text = build_hub.human_sentence({"answered": 40, "agree": 39, "alpha": 0.97})
        self.assertTrue(text.startswith("a native speaker answered 40"))


if __name__ == "__main__":
    unittest.main()
