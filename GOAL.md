# Backup consolidation — 2026-11-01

Preserve reproducibility while removing the nine temporary backup branches.

- Keep model cards, source, configs, task definitions, seeds and published data
  byte-identical on master. Restore the original standalone ipad_lab.html.
- Preserve historical design documents and README versions in Drive, with a
  per-file Git-blob manifest and a verified ZIP SHA-256.
- Keep all nine original heads reachable through master and immutable tags.
- Delete only the nine audited backup/2026-10-31/* branches after verification.
- Retain Claude, endophalon3, archives, iPad Lab, experiment and preserve branches.
- No retraining, experiment PR merge, force push or history rewriting.
- Audit and document archive: docs/integration/2026-11-01.md.

The preceding six-model integration remains documented in
docs/integration/2026-10-31.md.
