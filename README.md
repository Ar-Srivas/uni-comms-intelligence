## University Communications Intelligence

### Reproducible preprocessing

The raw dataset in `data/raw/` is an immutable input.  The preprocessing entry
point writes only to `data/processed/`:

```powershell
python scripts/preprocess.py
```

The command refuses to replace existing aggregate outputs.  Use `--force` to
regenerate them deterministically.  By default the final NLP dataset contains
only documents detected as English.  The master extracted, cleaned, quality,
and annotation outputs always retain every source PDF.  Use
`--include-non-english` to make non-English documents eligible for the final
dataset (documents with failed extraction or unresolved quality issues remain
out of it).

`scripts/preprocess.py --help` describes the optional input/output locations.
The quality report documents its thresholds and every exclusion/review reason.
