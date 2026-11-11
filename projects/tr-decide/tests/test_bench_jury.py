import json
import tempfile
import unittest
from pathlib import Path

from bench import jury


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def item(i: int, gold: int = 0) -> dict:
    return {"id": f"x-{i}", "state": "s", "question": "q", "options": ["a", "b", "c"],
            "option_descriptions": ["da", "db", "dc"], "gold": gold, "rationale": "r"}


def r1(i: int, answer: int, **kw) -> dict:
    return {"id": f"x-{i}", "answer": answer, "confidence": 0.9, "ambiguous": False, "natural": 5, "note": "", **kw}


def r2(i: int, verdict: str, correct: bool = True, unique: bool = True) -> dict:
    return {"id": f"x-{i}", "gold_correct": correct, "gold_unique": unique, "fix": "", "verdict": verdict}


class JuryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.saved = (jury.ROOT, jury.NATIVE, jury.JURY)
        jury.ROOT, jury.NATIVE, jury.JURY = root, root / "native", root / "jury"
        write(jury.NATIVE / "batch_b1.jsonl", [item(0), item(1), item(2), item(3)])

    def tearDown(self):
        jury.ROOT, jury.NATIVE, jury.JURY = self.saved
        self.tmp.cleanup()

    def round1(self, wrong: dict[int, int] | None = None, skip: tuple[str, int] | None = None):
        wrong = wrong or {}
        for j in jury.JUDGES:
            rows = [r1(i, 1 if wrong.get(i, 0) and j in ("gpt", "grok")[: wrong[i]] else 0) for i in range(4)]
            if skip and skip[0] == j:
                rows = [r for r in rows if r["id"] != f"x-{skip[1]}"]
            write(jury.JURY / f"r1_b1_{j}.jsonl", rows)

    def test_missing_round1_answer_is_an_error(self):
        self.round1(skip=("grok", 2))
        with self.assertRaises(jury.Incomplete):
            jury.review("b1")
        with self.assertRaises(jury.Incomplete):
            jury.final()

    def test_one_wrong_vote_sends_to_review(self):
        self.round1(wrong={1: 1})
        jury.review("b1")
        rows = jury.load(jury.JURY / "review_b1.jsonl")
        self.assertEqual([r["id"] for r in rows], ["x-1"])

    def test_unknown_ids_in_an_answer_file_are_an_error(self):
        self.round1()
        rows = jury.load(jury.JURY / "r1_b1_gemini.jsonl") + [r1(7, 0)]
        write(jury.JURY / "r1_b1_gemini.jsonl", rows)
        with self.assertRaises(jury.Incomplete):
            jury.review("b1")

    def test_a_note_alone_sends_to_review(self):
        self.round1()
        rows = jury.load(jury.JURY / "r1_b1_muse.jsonl")
        rows[2]["note"] = "Soru metni yönergeye atıf yapıyor ama cevap seyahat planına bağlı."
        write(jury.JURY / "r1_b1_muse.jsonl", rows)
        jury.review("b1")
        self.assertEqual([r["id"] for r in jury.load(jury.JURY / "review_b1.jsonl")], ["x-2"])

    def test_final_rules(self):
        self.round1(wrong={1: 1, 2: 2, 3: 1})
        verdicts = {1: ["keep", "keep", "keep", "drop"],          # kept
                    2: ["keep", "keep", "keep", "keep"],          # three keeps are inconsistent below -> dropped
                    3: ["fix", "fix", "keep", "drop"]}            # fixed and rechecked
        for n, j in enumerate(jury.JUDGES):
            rows = [r2(1, verdicts[1][n]), r2(3, verdicts[3][n]),
                    r2(2, verdicts[2][n], unique=n == 0)]         # only one reviewer says the gold is unique
            write(jury.JURY / f"r2_b1_{j}.jsonl", rows)
        with self.assertRaises(jury.Incomplete):                  # item 3 needs a hand fix first
            jury.final()
        write(jury.JURY / "fixed_b1.jsonl", [{**item(3, gold=2), "state": "s2"}])
        for j in jury.RECHECK_JUDGES:
            write(jury.JURY / f"r3_b1_{j}.jsonl", [r1(3, 2)])
        jury.final()
        kept = {r["id"]: r for r in jury.load(jury.ROOT / "native.jsonl")}
        self.assertEqual(sorted(kept), ["x-0", "x-1", "x-3"])
        self.assertEqual(kept["x-0"]["review"], "undisputed")
        self.assertEqual(kept["x-1"]["review"], "kept-after-review")
        self.assertEqual((kept["x-3"]["review"], kept["x-3"]["gold"], kept["x-3"]["state"]),
                         ("fixed-and-rechecked", 2, "s2"))

    def test_failed_recheck_drops_the_fixed_item(self):
        self.round1(wrong={3: 1})
        for j in jury.JUDGES:
            write(jury.JURY / f"r2_b1_{j}.jsonl", [r2(3, "fix")])
        write(jury.JURY / "fixed_b1.jsonl", [item(3, gold=2)])
        write(jury.JURY / "r3_b1_gpt.jsonl", [r1(3, 2)])
        write(jury.JURY / "r3_b1_gemini.jsonl", [r1(3, 1)])
        jury.final()
        self.assertEqual(sorted(r["id"] for r in jury.load(jury.ROOT / "native.jsonl")), ["x-0", "x-1", "x-2"])


if __name__ == "__main__":
    unittest.main()
