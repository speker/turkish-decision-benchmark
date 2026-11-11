import datetime as dt
import random
import re
import unittest
from decimal import Decimal

from bench.prog import (AMBALAJ_ACIK, AYLAR, FATURA_YOK, IADE_SECENEK, TATIL, YARIM_GUN, binlik, build, duyum_cifti,
                        gizli_talimat, gun_hesabi, iade_kurali, is_gunu, is_gunu_ekle, kdv_indirim, olumsuzluk_seti, para,
                        q2, sayi_bicim, tam_is_gunu, tarih, tarih_vade, uzun_giris, yaziyla)


def parse_tarih(s: str) -> dt.date:
    d, m, y = s.split()
    return dt.date(int(y), AYLAR.index(m) + 1, int(d))


def parse_para(s: str) -> Decimal:
    return Decimal(s.removesuffix(" TL").replace(".", "").replace(",", "."))


class ProgTest(unittest.TestCase):
    def test_formats(self):
        self.assertEqual(para(Decimal("1250.5")), "1.250,50 TL")
        self.assertEqual(para(Decimal("999")), "999,00 TL")
        self.assertEqual(para(Decimal("1234567.891")), "1.234.567,89 TL")
        self.assertEqual(binlik(12500), "12.500")
        self.assertEqual(tarih(dt.date(2026, 4, 23)), "23 Nisan 2026")

    def test_yaziyla(self):
        self.assertEqual(yaziyla(1000), "bin")
        self.assertEqual(yaziyla(1500), "bin beş yüz")
        self.assertEqual(yaziyla(12500), "on iki bin beş yüz")
        self.assertEqual(yaziyla(100), "yüz")
        self.assertEqual(yaziyla(2026), "iki bin yirmi altı")
        self.assertEqual(yaziyla(999999), "dokuz yüz doksan dokuz bin dokuz yüz doksan dokuz")

    def test_holidays(self):
        # Wednesday 22 April 2026 + 2 business days: Thursday 23 April is a holiday, so Friday 24 and Monday 27.
        self.assertEqual(is_gunu_ekle(dt.date(2026, 4, 22), 2), dt.date(2026, 4, 27))
        # Tuesday 17 March 2026 + 3: 18, 19 (eve, half day), then Ramazan Bayramı 20-22 March, so Monday 23 March.
        self.assertEqual(is_gunu_ekle(dt.date(2026, 3, 17), 3), dt.date(2026, 3, 23))
        self.assertIn(dt.date(2027, 5, 19), TATIL)
        self.assertIn(dt.date(2026, 10, 28), YARIM_GUN)

    def test_tarih_vade_due_date_is_full_business_day(self):
        rng = random.Random(1)
        for i in range(300):
            it = tarih_vade(rng, i)
            due = parse_tarih(it["options"][it["gold"]])
            start = parse_tarih(re.search(r"faturası (.+?) tarihinde", it["state"]).group(1))
            days = int(re.search(r"itibaren (\d+) gündür", it["state"]).group(1))
            self.assertEqual(due, start + dt.timedelta(days=days))
            self.assertTrue(tam_is_gunu(due), it)

    def test_gun_hesabi(self):
        rng = random.Random(2)
        for i in range(200):
            it = gun_hesabi(rng, i)
            m = re.search(r"randevunuz (.+?) (\w+) günü", it["state"])
            d = parse_tarih(m.group(1))
            self.assertEqual(it["options"][d.weekday()], m.group(2))
            k = int(re.search(r"randevudan (\d+) gün", it["state"]).group(1))
            target = d + dt.timedelta(days=k if it["variant"] == "sonra" else -k)
            self.assertEqual(it["gold"], target.weekday())

    def test_is_gunu_window_and_listed_holidays(self):
        rng = random.Random(3)
        for i in range(300):
            it = is_gunu(rng, i)
            start = parse_tarih(re.search(r"Başvurunuz (.+?) \w+ günü alındı", it["state"]).group(1))
            k = int(re.search(r"izleyen (\d+) iş günü", it["state"]).group(1))
            end = parse_tarih(it["options"][it["gold"]])
            self.assertEqual(end, is_gunu_ekle(start, k))
            self.assertTrue(tam_is_gunu(start))
            window = [start + dt.timedelta(days=j) for j in range((end - start).days + 1)]
            self.assertFalse(YARIM_GUN.intersection(window), it)
            holidays = [d for d in window if d in TATIL and d > start]
            self.assertEqual("resmî tatil yoktur" in it["state"], not holidays, it)

    def test_kdv_unambiguous_rounding_and_distractors(self):
        rng = random.Random(4)
        for i in range(300):
            it = kdv_indirim(rng, i)
            base = parse_para(re.search(r"birim fiyatı (.+? TL)", it["state"]).group(1))
            kdv = int(re.search(r"KDV oranı %(\d+)", it["state"]).group(1))
            m = re.search(r"%(\d+) indirim", it["state"])
            ind = int(m.group(1)) if m else 0
            after = base * (100 - ind) / 100
            gold = parse_para(it["options"][it["gold"]])
            self.assertEqual(gold, q2(after * (100 + kdv) / 100))
            self.assertEqual(gold, q2(q2(after) + q2(q2(after) * kdv / 100)))
            for j, o in enumerate(it["options"]):
                if j != it["gold"]:
                    self.assertGreaterEqual(abs(parse_para(o) - gold), Decimal("0.05"))

    def test_sayi_bicim_gold_by_variant(self):
        rng = random.Random(5)
        golds = set()
        for i in range(150):
            it = sayi_bicim(rng, i)
            golds.add(it["gold"])
            if it["variant"] in ("esit-bicim", "yaziyla-esit"):
                self.assertEqual(it["gold"], 2, it)
            else:
                self.assertIn(it["gold"], (0, 1), it)
            m = re.search(r"birinci kalem (.+), ikinci kalem (.+)\.$", it["state"])
            a, b = (read_amount(x) for x in m.groups())
            self.assertEqual(it["gold"], 2 if a == b else 0 if a > b else 1, it["state"])
        self.assertEqual(golds, {0, 1, 2})

    def test_olumsuzluk_sets_are_minimal(self):
        rng = random.Random(6)
        for i in range(30):
            s = olumsuzluk_seti(rng, i)
            self.assertEqual([x["gold"] for x in s], [1, 0, 1, 1, 0])
            self.assertEqual(len({x["state"].split(".")[0] for x in s}), 1)  # same "Merhaba, ben X"
            self.assertEqual(len({x["question"] for x in s}), 1)
            self.assertEqual(len({x["pair_id"] for x in s}), 1)
            self.assertNotIn("mez mi", " ".join(x["state"] for x in s))
            self.assertNotIn("maz mı", " ".join(x["state"] for x in s))

    def test_duyum_pairs_share_the_event(self):
        rng = random.Random(7)
        for i in range(24):
            di, mis = duyum_cifti(rng, i)
            self.assertEqual((di["gold"], mis["gold"]), (0, 1))
            self.assertEqual(di["state"].split()[:2], mis["state"].split()[:2])
            self.assertEqual(di["pair_id"], mis["pair_id"])
            self.assertRegex(mis["state"], r"m[ıiuü]ş\.")

    def test_iade_missing_fact_is_undecidable_only_when_nothing_else_fails(self):
        rng = random.Random(8)
        for i in range(400):
            eksik = rng.choice([None, "gun", "fatura", "ambalaj"])
            it = iade_kurali(rng, i, eksik=eksik)
            s = it["state"].split("7 gündür. ", 1)[1]
            fails = any(p in s for p in FATURA_YOK + AMBALAJ_ACIK)
            if "gün sonra" in s:
                days = int(s.split("teslimden ")[1].split(" gün")[0])
                limit = 7 if "elektronik kategorisinde" in s else 14
                self.assertNotEqual(days, limit)
                fails = fails or days > limit
            expected = 1 if fails else 0 if eksik is None else 2
            self.assertEqual(it["gold"], expected, s)
            self.assertEqual(it["options"], IADE_SECENEK)

    def test_planted_instruction_names_a_wrong_option(self):
        rng = random.Random(9)
        bases = [x for i in range(20) for x in olumsuzluk_seti(rng, i)] + [iade_kurali(rng, i) for i in range(50)]
        for n, b in enumerate(bases):
            it = gizli_talimat(rng, b, n)
            label = re.search(r"'([^']+)'", it["state"]).group(1)
            self.assertIn(label, it["options"])
            self.assertNotEqual(label, it["options"][it["gold"]])
            self.assertEqual(it["gold"], b["gold"])
            self.assertNotIn("..", it["state"])
            self.assertNotIn("?.", it["state"])

    def test_build_has_no_repeated_item_and_pairs_resolve(self):
        items = build()
        keys = [(x["state"], x["question"]) for x in items]
        self.assertEqual(len(keys), len(set(keys)))
        ids = {x["id"] for x in items}
        self.assertEqual(len(ids), len(items))
        for x in items:
            if x["family"] in ("gizli-talimat", "uzun-giris"):
                self.assertIn(x["pair_id"], ids)

    def test_long_prefix_never_repeats_a_sentence(self):
        rng = random.Random(10)
        filler = [f"Cümle numarası {w} bülten için yazıldı ve tekrar etmiyor." for w in yaziyla_list(300)]
        for n in range(20):
            base = iade_kurali(rng, n)
            it = uzun_giris(rng, base, n, rng.choice([150, 400, 900]), filler)
            body = it["state"].split("\n\nİade başvurusu\n")[0]
            sents = re.findall(r"[^.]+\.", body.removeprefix("Aylık bülten\n\n"))
            self.assertEqual(len(sents), len(set(sents)))
            self.assertTrue(it["state"].endswith(base["state"]))


def yaziyla_list(n: int) -> list[str]:
    return [yaziyla(i) for i in range(1, n + 1)]


BIR = ["bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"]
ON = ["on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"]


def read_amount(text: str) -> Decimal:
    """An amount as a Turkish reader reads it: "12.345 TL", "12.345,60 TL", "12345,60 TL" or "yazıyla ... lira"."""
    if text.startswith("yazıyla "):
        total = current = 0
        for w in text.removeprefix("yazıyla ").removesuffix(" lira").split():
            if w in BIR:
                current += BIR.index(w) + 1
            elif w in ON:
                current += 10 * (ON.index(w) + 1)
            elif w == "yüz":
                current = (current or 1) * 100
            elif w == "bin":
                total, current = total + (current or 1) * 1000, 0
            else:
                raise ValueError(w)
        return Decimal(total + current)
    return parse_para(text)


if __name__ == "__main__":
    unittest.main()
