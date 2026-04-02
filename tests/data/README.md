# Test Fixtures

This directory contains reproducible local restoration fixtures for `veriedit`.

- `sample_input.png` and `sample_reference.png` are the seed images.
- `manifest.json` lists generated source/reference pairs and their intended prompts.
- The derived pairs are created by `scripts/generate_test_pairs.py` so they can be regenerated instead of hand-edited.
