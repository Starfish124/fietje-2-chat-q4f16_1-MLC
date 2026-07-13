# fietje-2-chat — MLC/WebLLM build (q4f16_1)

In-browser (WebGPU) build van [BramVanroy/fietje-2-chat](https://huggingface.co/BramVanroy/fietje-2-chat),
het Nederlandstalige 2,7B-model gebaseerd op microsoft/phi-2 en doorgetraind op
28 miljard Nederlandse tokens. Alle credits voor het model: Bram Vanroy (MIT-licentie).

Deze repo bevat uitsluitend de naar MLC-formaat geconverteerde gewichten
(quantisatie `q4f16_1`), zodat het model met [WebLLM](https://github.com/mlc-ai/web-llm)
in de browser draait op de voorgecompileerde `phi-2` runtime. Daarvoor is de
vocab gepadded van 50297 naar 51200 (identiek aan de originele phi-2-lay-out);
de extra rijen zijn nul en worden nooit gesampled.

De conversie gebeurt volledig in GitHub Actions: zie
`.github/workflows/convert.yml` en `pad_vocab.py`.

Gebruik in WebLLM:

```js
const appConfig = {
  model_list: [{
    model: "https://raw.githubusercontent.com/Starfish124/fietje-2-chat-q4f16_1-MLC/main/weights",
    model_id: "fietje-2-chat-q4f16_1-MLC",
    model_lib: "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/v0_2_84/phi-2-q4f16_1_cs1k-webgpu.wasm",
    vram_required_MB: 3054,
    required_features: ["shader-f16"],
    overrides: { context_window_size: 2048 }
  }]
};
```

Gemaakt voor de CPI Marketing Motor (Nederlandstalige LinkedIn-postgenerator in de browser).
