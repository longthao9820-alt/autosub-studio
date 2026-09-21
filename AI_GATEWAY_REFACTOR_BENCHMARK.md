# AI Gateway refactor benchmark

Measured on 2026-09-21 with the same 79.658-second Chinese hardsub video and the configured AI Gateway roles.

## Subtitle extraction (`sub`)

| Path | Time | Sampled frames | AI images | AI requests | Upload | Cues | Retries |
|---|---:|---:|---:|---:|---:|---:|---:|
| Local PP-OCRv4 NTS (existing app task) | ~74 s | 996 | — | — | — | 24 | — |
| Old API implementation | 360.61 s | 1,046 | 820 | 104 | not recorded | 23 | 0 |
| New `sub` implementation, final 5 fps profile, thinking `high` | 54.98 s | 419 | 138 | 10 | 1.68 MiB | 23 | 0 |

The new image-fallback architecture reduced AI requests by 90.4% and total extraction time by about 6.6×. The new run produced the same 23 merged subtitle cues as the old AI run. The final 5 fps profile was retained instead of the slightly faster 3 fps experiment so short captions down to roughly 200 ms remain sampleable. The Gateway adapter currently exposes text and multi-image Chat Completions only; no video/media endpoint was invented.

## Subtitle translation (`prime`)

Test input: the same 23-cue source subtitle, 100 source characters, Chinese to Vietnamese, no missing or structurally invalid cues.

| Path | Thinking | Requests | Time | Missing/empty cues |
|---|---:|---:|---:|---:|
| Previous fixed batch size 8 | medium | 3 sequential | 31.67 s | 0 |
| New whole-file fast path | medium | 1 | 58.11 s | 0 |
| New whole-file fast path | low | 1 | 40.11 s | 0 |
| New whole-file fast path | none | 1 | 37.00 s | 0 |

For this unusually small file, reducing three requests to one did not reduce wall time: Gateway/model output latency dominated. The refactor still removes fixed small-batch fragmentation, preserves full-file context, validates IDs, checkpoints completed work, and repairs only failed IDs/chunks. Default `prime` thinking is now `low`; larger representative SRTs should be benchmarked before claiming a translation speedup.
