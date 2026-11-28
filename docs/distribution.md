# Downloading checkpoints

Run `hf download` from the root of the evaluation repository. The Hugging Face CLI preserves the directory layout expected by `evaluate.py`. [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli)

For one RLBench checkpoint, select the demonstration count in the filename.

```bash
hf download mingchens/EnergyBimanual --local-dir . \
  --include "checkpoints/*/energybimanual/5demo.pth"
```

For RoboTwin DP and DP3, select `dp` or `dp3` and the corresponding `.pt` file.

```bash
hf download mingchens/EnergyBimanual --local-dir . \
  --include "checkpoints/*/dp3/10demo.pt"
```

For RoboTwin π₀ and π₀.₅, select the complete Orbax directory, including its metadata, parameter shards and normalization statistics.

```bash
hf download mingchens/EnergyBimanual --local-dir . \
  --include "checkpoints/*/pi05/20demo/*"
```

Verify the files you downloaded with GNU coreutils.

```bash
sha256sum --check --ignore-missing CHECKSUMS.sha256
```
