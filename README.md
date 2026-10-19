# strata-879-repro

A reproduction for [Niko1221/Strata#879](https://github.com/Niko1221/Strata/issues/879): a real agent request
(18442 prompt tokens: a system prompt plus 48 tool schemas) makes the engine's logits non-finite twice in a row,
so the server answers 503 `the model's output became non-finite (NaN) twice in a row (#879)`.

The same request answers normally with `STRATA_PREFILL_MMQ=0`. The NaN is born in layer 1's MoE gate/up on the
MMQ prompt path, in all three prefill chunks.

## What is here

| path | what it is |
| --- | --- |
| `request/request.json` | the captured request body, unmodified (system prompt, 3 more messages, 48 tool schemas, `stream`, `max_completion_tokens` 32768) |
| `prompt/system.txt` | message 0 of that body, the 6745-character system text, for slicing |
| `replay.py` | replays the body against a running server, with switches that reshape it (`--drop-tools`, `--keep-msg`, `--system-chars`, `--tool-limit`, `--tools-scale`) |
| `probe.py` | sends one small chat request in a new conversation each time, for the token-level reducer below |
| `checks/proxy.py` | a one-hop logging proxy, how the body was captured |
| `checks/emb_scan.py` | scans the native GGUF's `token_embd.weight` (IQ4_XS) row by row for non-finite or extreme values |
| `checks/ple_rows.py` | reads the PLE n-gram rows (`per_layer_token_embd.weight`, IQ4_NL) a token bigram selects |

## Environment it was measured on

- engine 0.1.45 (release exe), Windows, RTX 3080 20 GB, 96 GB RAM
- model `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S`
- engine args: `--max-context 524288 --rope-scaling yarn --rope-scale 2 --kv int8 --kv-resident 32768 --spec 4 --spec-min-p 0.5 --mtp ... --prefill auto --expert-cache auto`
- server: `python serve/server.py --engine strata --config strata-iq3_s.json --port 8080`

## Reproduce

Start Strata, wait until the model is loaded, then:

```
python replay.py request/request.json --port 8080
```

Expected with the default MMQ prompt path:

```
-> tools=48 msg=4 bytes=80116
status=200 25.9s finish=None usage=None chars=0
ERROR: {"type": "server_error", "message": "the model's output became non-finite (NaN) twice in a row (#879); the engine has dropped its cached state - send the request again"}
```

The server's own window says:

```
strata serve: non-finite logits (#879) in a prompt window at position 18437 (row 0): its tokens are not emitted, the request ends (DONE nonfinite)
strata serve: prompt 18442 tokens = 0 reused + 18437 of 18442 read in 14475 ms (1273.8 tok/s), 0 generated in 0 ms (0.0 tok/s), drafts accepted 0 of 0, 2 checkpoints (cancelled)
```

Stop the server, start it with `STRATA_PREFILL_MMQ=0`, and send the same body again:

```
status=200 27.2s finish=stop usage={'prompt_tokens': 18442, 'completion_tokens': 238, ...} chars=352
strata serve: prompt 18442 tokens = 0 reused + 18442 read in 21633 ms (852.5 tok/s), 238 generated in 5214 ms (45.6 tok/s)
```

So the two runs read the same 18442 tokens, at 1273.8 vs 852.5 tok/s, and only the first one ends non-finite.

Windows, one shell:

```
set STRATA_PREFILL_MMQ=0
.venv\Scripts\python.exe serve\server.py --engine strata --config strata-iq3_s.json --port 8080
```

## Where it dies

`STRATA_DBG_NAN=1` names it (it changes timing, so it was used to locate the failure, not to measure anything):

```
strata dbg: layer 1 (mmq 1, types 22/42, 476 experts): non-finite GU 89024000 H 44512000 Dm 178048000 bo 17804800 of T 8192
strata dbg: layer 1 (mmq 1, types 22/42, 10 experts): non-finite GU 94860800 H 47430400 Dm 189721600 bo 18972160 of T 7411
strata dbg: layer 1 (mmq 1, types 22/42, 10 experts): non-finite GU 36275200 H 18137600 Dm 72550400 bo 7255040 of T 2834
strata dbg: prompt end: last residual row 10240 non-finite (max |x| 0); PLE history 0 non-finite (max |x| 14.8); GDN state 0 0 non-finite (max |x| 1.86)
strata dbg: verify window at position 18437, row 0: 248320 of 248320 logits non-finite (token out 0)
```

Three chunks (8192 + 7411 + 2834 = 18437), the whole gate/up and down output of layer 1, at `mmq 1`, while PLE
and the GDN state are finite.

## What was ruled out

- **Model data.** `checks/emb_scan.py` over all 248320 rows of `token_embd.weight`: 0 non-finite block scales,
  per-row max |v| median 0.0368, p99 0.0787, p99.9 0.1211, max 0.2381. Row 96202 is normal (0.0308).
- **PLE rows.** `checks/ple_rows.py` for the 16 rows the reducer's bigram selects: max |v| 0.019 to 0.033, block
  scales about 1e-4, like their neighbours.
- **KV int8.** `--kv int8` is in both runs above, so its quantizer is not the difference.
- **The conversation cache.** The first pass reports `0 reused`, and the guard drops the session before the retry.
- **"the prompt is just long".** An 18380-token plain-text prompt and a 17877-token prompt with 72 tool schemas
  both answer with MMQ on.

## The token-level reducer

Bisecting the body with MMQ on: dropping all 48 tools still fails, and keeping only the system message plus
`你好` still fails. The smallest reducer found is a system text ending in `传` preceded by a space token (ids 220,
96202):

```
python probe.py --system "x 传"                                        -> nonfinite
python probe.py --system " 传"                                         -> nonfinite
python probe.py --system "传"                                          -> ok
python probe.py --system " 转"                                         -> ok
python probe.py --system-file prompt/system.txt --system-chars 3400    -> nonfinite
python probe.py --system-file prompt/system.txt --system-chars 3399    -> ok
```

The 3400-character slice ends with `传`; the 3399-character one ends one character earlier. Both figures are from
a warm engine, and the reducer is what the bisect narrowed to, not the whole story.

This reducer only fires while the engine is warm (a cached prefix); from a cold engine the same 58-token prompt
passes. Treat `request/request.json` as the reliable reproducer and this as a lead.

`probe.py` puts a rotating single-token prefix in front of every request. Each character in its marker set is one
token in this model's vocabulary, so no two requests share a cached prefix while the reported `prompt_tokens`
stays the same. `传` is left out of the set on purpose.

## Notes

- `request/request.json` is exactly what the client sent: its own system prompt, the conversation, and the 48 tool
  schemas. It contains no credentials. It does contain the client's local paths, published here on purpose so the
  prompt tokenizes to the same 18442 tokens.
- The two candidate sites for the real fix (an fp16 scale in the MMQ q8_1 quantizer, and the MMQ prompt path's
  borrowed expert slots) are written up in the issue. Neither is confirmed: there was no build environment here
  (nvcc 13.4 present, cmake and MSVC missing), so nothing in the engine was changed to test them.
