# Models Metadata

This directory contains metadata artifacts, evaluation benchmarks, class schema mappings, and architecture notes for registered models.

## Metadata Guidelines
- Class maps must accurately reflect zero-indexed integer IDs to class names.
- Inference resolution benchmarks should document latency vs mAP tradeoffs (e.g. 640px vs 1280px on GPU/CPU).
- Small-object sensitivity metrics (recall on targets < 32×32 pixels) should be logged here following custom training.
