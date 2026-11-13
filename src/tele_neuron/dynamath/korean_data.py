"""Pinned Song Ys acquisition, immutable grouped pair split and fixed tokenizer."""
import argparse
import collections
import csv
import datetime
import hashlib
import io
import json
from pathlib import Path
import unicodedata as ud
import urllib.request

from .training import dump

SOURCE = "https://github.com/songys/Chatbot_data"
COMMIT = "4cf20d13fc46f5037fd1c531cd566e2dd9f72974"
BLOB = "4cc2bfa9a124991dc5ca80f5da1786f4162f46e1"
USER, BOT, EOS, SILENCE, UNK = "\ue000\ue001\ue002\ue003\ue004"
CONTROLS = USER + BOT + EOS + SILENCE + UNK


def sha(data):
    return hashlib.sha256(data).hexdigest()


def normalize(text):
    return ud.normalize("NFC", text)


def question_key(text):
    return "".join(c for c in normalize(text).casefold() if c.isalnum())


class Tokenizer:
    def __init__(self, vocabulary):
        if not vocabulary.startswith(CONTROLS) or len(set(vocabulary)) != len(vocabulary):
            raise ValueError("invalid fixed vocabulary")
        self.vocabulary = vocabulary
        self.ids = {c: i for i, c in enumerate(vocabulary)}

    @classmethod
    def build(cls, rows):
        chars = set("".join(ud.normalize("NFD", normalize(r[k]))
                            for r in rows for k in ("Q", "A")))
        if chars & set(CONTROLS):
            raise ValueError("corpus contains reserved controls")
        return cls(CONTROLS + "".join(sorted(chars)))

    def encode(self, text):
        return [self.ids.get(c, self.ids[UNK]) for c in ud.normalize("NFD", normalize(text))]

    def decode(self, ids):
        # Preserve unattached Jamo; never repair model output into an answer.
        chars = [self.vocabulary[i] for i in ids]
        return normalize("".join("�" if c == UNK else c for c in chars if c not in CONTROLS or c == UNK))

    def pairs(self, row):
        q, a = self.encode(row["Q"]), self.encode(row["A"])
        sil, eos = self.ids[SILENCE], self.ids[EOS]
        inputs = [self.ids[USER]] + q + [self.ids[BOT]] + a + [eos]
        targets = [sil] * (1 + len(q)) + a + [eos, sil]
        assert len(inputs) == len(targets)
        return inputs, targets


def grouped_split(rows, seed=43):
    groups = collections.defaultdict(list)
    for r in rows:
        groups[question_key(r["Q"])].append(r["id"])
    split = {s: [] for s in ("train", "validation", "test")}
    # Hash per normalized group: independent of CSV ordering; expected 80/10/10.
    for key, ids in groups.items():
        bucket = int(sha((str(seed) + ":" + key).encode())[:16], 16) % 100
        name = "train" if bucket < 80 else "validation" if bucket < 90 else "test"
        split[name].extend(ids)
    return {s: sorted(ids) for s, ids in split.items()}, groups


def fetch(name):
    url = f"https://raw.githubusercontent.com/songys/Chatbot_data/{COMMIT}/{name}"
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def prepare(directory, upstream=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    raw = (Path(upstream) / "ChatbotData.csv").read_bytes() if upstream else fetch("ChatbotData.csv")
    license_data = (Path(upstream) / "LICENSE").read_bytes() if upstream else fetch("LICENSE")
    blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    if blob != BLOB:
        raise ValueError("upstream CSV differs from predeclared source pin")
    rows, invalid, label_cleanups, normalization_changes = [], [], [], {}
    for i, row in enumerate(csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))):
        if not row.get("Q", "").strip() or not row.get("A", "").strip() or row.get("label", "").strip() not in ("0", "1", "2"):
            invalid.append(i)
            continue
        if row["label"] != row["label"].strip():
            label_cleanups.append(i)
        normalization_changes[i] = sum(normalize(row[k]) != row[k] for k in ('Q', 'A'))
        rows.append({"id": i, "Q": normalize(row["Q"]), "A": normalize(row["A"]), "label": row["label"].strip()})
    splits, groups = grouped_split(rows)
    by_id = {r["id"]: r for r in rows}
    tok = Tokenizer.build([by_id[i] for i in splits["train"]])
    manifest = {"source_commit": COMMIT, "csv_sha256": sha(raw), "seed": 43,
                "group_policy": "NFC, casefold, alphanumeric-only; hash(seed:key)%100",
                "splits": splits,
                "group_sha256": {str(r["id"]): sha(question_key(r["Q"]).encode()) for r in rows}}
    manifest_path = directory / "split-manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("immutable split manifest mismatch; refusing overwrite")
    dump(manifest_path, manifest)
    dump(directory / "tokenizer.json", {"vocabulary": tok.vocabulary, "size": len(tok.vocabulary),
         "normalization": "NFC ingest then NFD; decode NFC; unknown codepoints -> UNK",
         "controls": {"USER": 0, "BOT": 1, "EOS": 2, "SILENCE": 3, "UNK": 4},
         "learned": False})
    report = {"credit": "Song Ys (songys)", "source": SOURCE, "commit": COMMIT,
              "csv_blob_sha": blob, "csv_sha256": sha(raw), "license_sha256": sha(license_data),
              "acquired_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "parsed_rows": len(rows) + len(invalid), "valid_rows": len(rows),
              "documented_approx_rows": 11876, "invalid_ids": invalid,
              "label_whitespace_cleaned_ids": label_cleanups,
              "duplicate_QA_excess": len(rows) - len({(r['Q'], r['A']) for r in rows}),
              "duplicate_Q_excess": len(rows) - len({r['Q'] for r in rows}),
              "normalized_question_groups": len(groups), "splits": {}}
    for name, ids in splits.items():
        subset = [by_id[i] for i in ids]
        encoded = [tok.encode(r[k]) for r in subset for k in ("Q", "A")]
        tokens = sum(map(len, encoded))
        unknown = sum(x == tok.ids[UNK] for seq in encoded for x in seq)
        losses = sum(tok.decode(tok.encode(r[k])) != normalize(r[k]) for r in subset for k in ("Q", "A"))
        report["splits"][name] = {"rows": len(ids), "labels": dict(collections.Counter(r['label'] for r in subset)),
          "text_tokens": tokens, "unknown_tokens": unknown, "unknown_rate": unknown / tokens,
          "roundtrip_failures": losses, "NFC_changes_from_raw": sum(normalization_changes[i] for i in ids),
          "unknown_codepoints": sorted({f'U+{ord(c):04X}' for r in subset for k in ('Q', 'A') for c in ud.normalize('NFD', r[k]) if c not in tok.ids}),
          "mean_QA_tokens": tokens / len(ids), "max_text_tokens": max(map(len, encoded)),
          "stream_positions": sum(len(tok.pairs(r)[0]) for r in subset),
          "normalized_characters": sum(len(r[k]) for r in subset for k in ("Q", "A"))}
    report['cross_split_overlap'] = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        a, b = [by_id[i] for i in splits[left]], [by_id[i] for i in splits[right]]
        report['cross_split_overlap'][left + ':' + right] = {
          'normalized_Q': len({question_key(r['Q']) for r in a} & {question_key(r['Q']) for r in b}),
          'exact_A': len({r['A'] for r in a} & {r['A'] for r in b}),
          'exact_QA': len({(r['Q'], r['A']) for r in a} & {(r['Q'], r['A']) for r in b})}
    dump(directory / "source-report.json", report)
    (directory / "songys-LICENSE.txt").write_bytes(license_data)
    (directory / "ChatbotData.csv").write_bytes(raw)
    return report


def load(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "split-manifest.json").read_text())
    raw = (directory / "ChatbotData.csv").read_bytes()
    if sha(raw) != manifest["csv_sha256"]:
        raise ValueError("CSV hash mismatch")
    rows = {i: {**r, "id": i, "Q": normalize(r['Q']), "A": normalize(r['A']), "label": r['label'].strip()}
            for i, r in enumerate(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))}
    tok = Tokenizer(json.loads((directory / "tokenizer.json").read_text())["vocabulary"])
    return {s: [rows[i] for i in ids] for s, ids in manifest["splits"].items()}, tok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default="data/dynamath/001a")
    ap.add_argument("--upstream", help="optional local pinned upstream checkout")
    args = ap.parse_args()
    print(json.dumps(prepare(args.output, args.upstream), indent=2, ensure_ascii=False))
