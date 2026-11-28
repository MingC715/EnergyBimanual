<div align="center">

# EnergyBimanual: A Unified Energy Framework for Unimanual-to-Bimanual Policy Composition

<div>
<a href="https://ieeexplore.ieee.org/author/37087008154">Weili&nbsp;Guan</a><sup>1 3</sup>,
Mingchen&nbsp;Song<sup>1 2</sup>,
<a href="https://homepage.hit.edu.cn/dengxiang">Xiang&nbsp;Deng</a><sup>1 3</sup>,
Jie&nbsp;Wei<sup>1</sup>,
<a href="https://scholar.google.com/citations?hl=en&amp;user=Awsue7sAAAAJ">Dongmei&nbsp;Jiang</a><sup>2</sup>,
<a href="https://scholar.google.com/citations?hl=en&amp;user=yywVMhUAAAAJ">Liqiang&nbsp;Nie</a><sup>1</sup>
</div>
<sup>1</sup>Harbin Institute of Technology (Shenzhen)&nbsp;&nbsp;
<sup>2</sup>Peng Cheng Laboratory&nbsp;&nbsp;
<sup>3</sup>Shenzhen Loop Area Institute
<br><br>
<a href="#manuscript-citation"><img src="https://img.shields.io/badge/Manuscript-Citation-blue" alt="Cite the EnergyBimanual manuscript"></a>
<a href="evaluate.py"><img src="https://img.shields.io/badge/Code-Evaluation-black" alt="Evaluation code"></a>
<a href="https://huggingface.co/mingchens/EnergyBimanual"><img src="https://img.shields.io/badge/Checkpoints-Hugging%20Face-yellow" alt="EnergyBimanual checkpoints on Hugging Face"></a>

</div>

## 📖 Introduction

During a handover, the receiving gripper must secure the object before the other lets go. When lifting a tray, both arms must support it throughout the lift. **EnergyBimanual** reuses pretrained single-arm policies and learns these different forms of coordination from a small set of bimanual demonstrations.

[![EnergyBimanual framework showing single-arm policy composition at the top, cross-arm interaction at lower left, and energy-aware denoising at lower right.](docs/images/energybimanual-framework.png)](docs/images/energybimanual-framework.png)

In **panel (a)**, each arm’s policy evaluates a proposed motion at sampled noise levels. A motion that better matches the policy’s predictions receives a lower energy. Adding the left and right energies gives a common score even when one arm uses a diffusion policy and the other uses flow matching.

**Bimanual Energy Interaction**, shown in **panel (b)**, adds a coordination term to that score. Each arm’s encoder takes its proposed action, its state, the visual observation, and the language instruction. Bidirectional cross-attention lets each arm account for the other’s motion. Smoothness and collision penalties discourage abrupt movements and collisions between the arms.

**Panel (c)** shows how the paper uses the joint energy to guide denoising. A lightweight controller reads the current energy and its change since the previous step, adjusts the step size, and decides when to stop. The underlying policy stays fixed and supplies the direction for each update.

This repository provides evaluation code and checkpoints for 13 bimanual RLBench tasks and six RoboTwin 2.0 tasks, with 5, 10, or 20 bimanual demonstrations per task.

## ⚙️ Environment Setup

Install the simulator and its assets, then install the requirements for the selected policy in that environment.

All RoboTwin 2.0 evaluations used the following environment.

- Python 3.11
- JAX/JAXLIB 0.5.3
- Torch 2.7.1
- Orbax Checkpoint 0.11.13
- NVIDIA 550 (driver)
- `openpi/.venv`

| Benchmark | Runtime dependencies | Policy requirements |
| --- | --- | --- |
| Bimanual RLBench | Python 3.10, PyTorch 2.2.0, torchvision 0.17.0, PerAct² RLBench, PyRep, CoppeliaSim | [EnergyBimanual](requirements/rlbench.txt) |
| RoboTwin 2.0 | SAPIEN 3.0.0b1, MPLib 0.2.1, cuRobo | [DP and DP3](requirements/diffusion.txt) · [π₀ and π₀.₅](requirements/openpi.txt) |

```bash
python -m pip install -r requirements/rlbench.txt
```

For RLBench, set `PERACT2_DATA_ROOT` to the 256 × 256 test split containing the task folders. CoppeliaSim needs a working display. [PerAct² installation](https://github.com/markusgrotz/peract_bimanual/blob/main/INSTALLATION.md) · [Test data](https://bimanual.github.io/)

For RoboTwin, set `ROBOTWIN_ROOT` to the directory containing `assets/objects/` and `assets/embodiments/`. SAPIEN uses Vulkan for rendering. [RoboTwin installation and assets](https://robotwin-platform.github.io/doc/index.html)

## 📦 Checkpoints

The model weights are being uploaded to Hugging Face. The entries below link to their SHA-256 checksums. [Hugging Face model repository](https://huggingface.co/mingchens/EnergyBimanual)

```bash
sha256sum -c CHECKSUMS.sha256
```

### Bimanual RLBench

| Demos per task | EnergyBimanual |
| ---: | --- |
| 5 | [energybimanual/5demo.pth](CHECKSUMS.sha256#L3) |
| 10 | [energybimanual/10demo.pth](CHECKSUMS.sha256#L1) |
| 20 | [energybimanual/20demo.pth](CHECKSUMS.sha256#L2) |

### RoboTwin 2.0

| Demos per task | DP + DP | DP3 + DP3 | π₀ + π₀ | π₀.₅ + π₀.₅ |
| ---: | --- | --- | --- | --- |
| 5 | [dp/5demo.pt](CHECKSUMS.sha256#L6) | [dp3/5demo.pt](CHECKSUMS.sha256#L9) | [pi0/5demo/](CHECKSUMS.sha256#L41-L55) | [pi05/5demo/](CHECKSUMS.sha256#L90-L105) |
| 10 | [dp/10demo.pt](CHECKSUMS.sha256#L4) | [dp3/10demo.pt](CHECKSUMS.sha256#L7) | [pi0/10demo/](CHECKSUMS.sha256#L10-L23) | [pi05/10demo/](CHECKSUMS.sha256#L56-L74) |
| 20 | [dp/20demo.pt](CHECKSUMS.sha256#L5) | [dp3/20demo.pt](CHECKSUMS.sha256#L8) | [pi0/20demo/](CHECKSUMS.sha256#L24-L40) | [pi05/20demo/](CHECKSUMS.sha256#L75-L89) |

DP and DP3 use Franka. The π policies use ALOHA and require the complete checkpoint directory, including normalization statistics.

## 🧪 Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py rlbench \
  --method energybimanual --demos 5 --task bimanual_push_box --seed 0

CUDA_VISIBLE_DEVICES=0 python evaluate.py robotwin \
  --method dp3 --demos 5 --task grab_roller --seed 0

CUDA_VISIBLE_DEVICES=1 python evaluate.py robotwin \
  --method pi0 --demos 5 --task place_burger_fries --seed 0
```

Use `--task all` for the full task list and `--demos` with 5, 10 or 20 to select the checkpoint. The optional ALOHA patch skips a missing planner target during the expert check and records skipped scene seeds.

```bash
git apply --check patches/aloha-expert-seeds.patch
git apply patches/aloha-expert-seeds.patch
```

```bash
python summarize_results.py rlbench --folder outputs/rlbench/energybimanual/5demo/seed0
python summarize_results.py robotwin --folder outputs/robotwin/dp3/5demo/seed0
```

## 🗂️ Project Structure

```text
evaluate.py                  Evaluation entry point
summarize_results.py         Task success rates
rlbench2/                    RLBench policies and evaluator
robotwin/                    RoboTwin policies, tasks and evaluator
requirements/                Policy dependencies
patches/                     Optional ALOHA expert patch
checkpoints/                 Model weights (separate download)
CHECKSUMS.sha256             Checkpoint checksums
licenses/                    Third-party licenses
```

## 🙏 Acknowledgement

We thank the PerAct² and RoboTwin teams for their simulators, tasks and evaluation assets. The policy implementations build on the work of the 3D FlowMatch Actor, Diffusion Policy, 3D Diffusion Policy and OpenPI authors. [PerAct²](https://bimanual.github.io/) · [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) · [3D FlowMatch Actor](https://github.com/nickgkan/3d_flowmatch_actor) · [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) · [3D Diffusion Policy](https://github.com/YanjieZe/3D-Diffusion-Policy) · [OpenPI](https://github.com/Physical-Intelligence/openpi)

<a name="manuscript-citation"></a>

## 📚 Citation

```bibtex
@misc{guan2026energybimanual,
  title  = {{EnergyBimanual}: A Unified Energy Framework for Unimanual-to-Bimanual Policy Composition},
  author = {Guan, Weili and Song, Mingchen and Deng, Xiang and Wei, Jie and Jiang, Dongmei and Nie, Liqiang},
  year   = {2026},
  note   = {Manuscript}
}
```
