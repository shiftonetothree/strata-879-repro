# strata-879-repro

Reproduction for [Niko1221/Strata#879](https://github.com/Niko1221/Strata/issues/879).

A real agent request (18442 prompt tokens: a system prompt plus 48 tool schemas) ends with

```
the model's output became non-finite (NaN) twice in a row (#879); the engine has dropped its cached state - send the request again
```

on the default MMQ prompt path, and answers normally with `STRATA_PREFILL_MMQ=0`. The two runs read the same
18442 tokens, at 1273.8 vs 852.5 tok/s, and only the first one ends non-finite.

## Environment

- Windows, NVIDIA RTX 3080 20 GB, 96 GB RAM
- engine 0.1.45 (release exe, CUDA 13.0 build)
- model `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S`

Engine launch parameters (these are this machine's paths):

```
strata.exe --serve --pack D:\Strata-data\packs\iq3_s
           --native D:\Strata-data\models\IQ3_S\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
           --ple-gguf D:\Strata-data\models\IQ3_S\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf
           --expert-profile D:\Strata\data\expert-profile.bin --expert-cache auto --prefill auto
           --spec 4 --spec-min-p 0.5 --mtp D:\Strata-data\mtp\rt
           --max-context 524288 --rope-scaling yarn --rope-scale 2
           --kv int8 --kv-resident 32768 --vision
```

Server command:

```
python serve/server.py --engine strata --config strata-iq3_s.json --port 8080
```

`strata-iq3_s.json` holds those same engine arguments, `--port 8080`, `--vision` and the tokenizer path.

## Reproduce

1. Start the server as above and wait until the model is loaded.
2. `python replay.py request/request.json --port 8080`

Expected: HTTP 200 with a stream that carries an error, and in the server's window

```
strata serve: non-finite logits (#879) in a prompt window at position 18437 (row 0): its tokens are not emitted, the request ends (DONE nonfinite)
strata serve: prompt 18442 tokens = 0 reused + 18437 of 18442 read in 14475 ms (1273.8 tok/s), 0 generated in 0 ms (0.0 tok/s), drafts accepted 0 of 0, 2 checkpoints (cancelled)
```

3. Start the server again with `STRATA_PREFILL_MMQ=0` (Windows: `set STRATA_PREFILL_MMQ=0` in the same shell) and
   send the same body:

```
status=200 27.2s finish=stop usage={'prompt_tokens': 18442, 'completion_tokens': 238, ...} chars=352
strata serve: prompt 18442 tokens = 0 reused + 18442 read in 21633 ms (852.5 tok/s), 238 generated in 5214 ms (45.6 tok/s)
```

## Files

- `request/request.json`: the captured body, unmodified. It is the prompt: the client's own system message, 3 more
  messages, and 48 tool schemas. No credentials; the client's local paths are left in so the prompt tokenizes to
  the same 18442 tokens.
- `replay.py`: sends that body and prints what came back.
