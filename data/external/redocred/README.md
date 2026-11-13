# External evaluation data: Re-DocRED

Downloaded from the dataset authors. The original `dev_revised.json` evaluation was followed by a supervised experiment using the separate training split; test data is never used to fit weights or select thresholds.

- Data: https://raw.githubusercontent.com/tonytan48/Re-DocRED/main/data/dev_revised.json
- Authors/repository: https://github.com/tonytan48/Re-DocRED
- Paper: Tan et al., *Revisiting DocRED - Addressing the False Negative Problem in Relation Extraction*, EMNLP 2022, https://arxiv.org/abs/2205.12696
- Development: all 500 documents, used first as external evidence and later for learned-candidate threshold selection. The original rule-only comparison used no training on this data.
- Training: `train_revised.json`, 3,053 source documents, SHA-256 `c40a137a1c57f3e2daaf5ebf511eaa1518f9305ffaa3052d46271921a91826f2`. One train/dev title overlap was excluded, leaving 3,052 training documents.
- Test: `test_revised.json`, 500 documents, downloaded only after the learned artifact/features/thresholds were frozen. SHA-256 `ea673b7ef91e16510d7eb88863d3903e47318dc5653ef361266f27ae3b84723c`. No title or normalized-text overlaps with train/dev were found. This split has now been evaluated and must not be described as a fresh unseen set for further tuning.
- Dataset SHA-256: `051ee1d057204a5d08ef5502beacdadf191245b5eaf0e29ec2c607cf002c016f`
- License: the authors' MIT license is preserved in `LICENSE`.
- Original relation names: https://huggingface.co/datasets/thunlp/docred/resolve/main/data/rel_info.json.gz (decompressed into `rel_info.json`).

The evaluation protocol is in `output/evaluation/protocol.md`. Frozen baseline/revised source and model hashes are in `output/evaluation/freeze_manifest.json`. These are Wikipedia-derived text documents, not a representative sample of corporate-finance PDFs. This benchmark does not evaluate OCR or page layout, and its scores are not official Re-DocRED leaderboard results.

The downloaded corpus and compressed relation-name archive are kept locally and
excluded from Git. The license, provenance notes and decompressed `rel_info.json`
remain versioned. To fetch all three corpus splits in PowerShell:

```powershell
foreach ($split in 'train', 'dev', 'test') {
    Invoke-WebRequest -Uri "https://raw.githubusercontent.com/tonytan48/Re-DocRED/main/data/${split}_revised.json" -OutFile "data/external/redocred/${split}_revised.json"
    Get-FileHash "data/external/redocred/${split}_revised.json" -Algorithm SHA256
}
```

Check the hash before comparing runs; the upstream `main` branch can change.
