# Color Matching Using Hypernetwork-Based Kolmogorov-Arnold Networks

[🇷🇺 Russian language (Русский язык)](README_RU.md)

Authors: Artem Nikonorov, Georgy Perevozchikov, Andrei Korepanov, Nancy Mehta, Mahmoud Afifi, Egor Ershov, Radu Timofte.

![abstract](figures/abstract.png)

### Abstract

We present ***cmKAN***, a versatile framework for color matching. Given an input image with colors from a source color distribution, our method effectively and accurately maps these colors to match a target color distribution in both **supervised** and **unsupervised** settings. Our framework leverages the spline capabilities of Kolmogorov-Arnold Networks (KANs) to model the color matching between source and target distributions. Specifically, we developed a hypernetwork that generates spatially varying weight maps to control the nonlinear splines of a KAN, enabling accurate color matching. As part of this work, we introduce a large-scale dataset of paired images captured by two distinct cameras to evaluate our method’s efficacy in matching colors produced by different cameras. We evaluated our approach across various color-matching tasks, including: (1) **raw-to-raw mapping**, where the source color distribution is in one camera’s raw color space and the target in another camera’s raw space; (2) **raw-to-sRGB mapping**, where the source color distribution is in a camera’s raw space and the target is in the display sRGB space, emulating the color rendering of a camera ISP; and (3) **sRGB-to-sRGB mapping**, where the goal is to transfer colors from a source sRGB space (e.g., produced by a source camera ISP) to a target sRGB space (e.g., from a different camera ISP). The results demonstrate that our method achieves state-of-the-art performance across these tasks while remaining lightweight compared to other color matching and transfer methods.

## Installation and Requirements

Create a conda (or python) environment, clone the repository and install the required packages:

```bash
# 1. Create an environment

conda create -n cmKAN python=3.10 pip
conda activate cmKAN
# or
python -m venv .venv
source .venv/bin/activate

# 2. Clone the repository
git clone https://github.com/gosha20777/cmKAN.git
cd cmKAN

# 3. Install the required packages
pip install -r requirements.txt
```

## Dataset

We introduce a large-scale [Volga2K dataset](https://huggingface.co/datasets/gosha20777/volga2k), featuring over 2000 well-aligned images captured with a Huawei P40 Pro. This device was selected for its distinct Quad-Bayer RGGB and RYYB camera sensors, which employ different image processing techniques. These sensor variations in color handling, sensitivity, and tone-mapping create a significant domain gap. Spanning four years and multiple locations, the dataset is designed to effectively evaluate color-matching methods.

You can find more details about the Volga2K dataset on [😡 Hugging Face 😡 official page](https://huggingface.co/datasets/gosha20777/volga2k). To download Volga2K, please use the following command:

```bash
huggingface-cli download gosha20777/volga2k --repo-type dataset --local-dir data/
```

## Pre-trained Models

We provide pre-trained models for the following tasks:

| Dataset | Task | Training Config | Checkpoint | 
| ------- | ---- | ---------------- | ---------- |
| [Volga2K](https://huggingface.co/datasets/gosha20777/volga2k) | sRGB-to-sRGB | [unsupervised](configs/) | [checkpoint](https://github.com/gosha20777/cmKAN/releases) |
| [Volga2K](https://huggingface.co/datasets/gosha20777/volga2k) | sRGB-to-sRGB | [supervised](configs/) | [checkpoint](https://github.com/gosha20777/cmKAN/releases) |
| [Volga2K](https://huggingface.co/datasets/gosha20777/volga2k) | sRGB-to-sRGB | [pair-based](configs/) | [checkpoint](https://github.com/gosha20777/cmKAN/releases) |
| [Adobe 5K](https://pan.baidu.com/share/init?surl=CsQRFsEPZCSjkT3Z1X_B1w) (password: `5fyk`)| sRGB-to-sRGB | [supervised](configs/) | [checkpoint](https://github.com/gosha20777/cmKAN/releases) |
| [Samsung2Iphone](https://github.com/mahmoudnafifi/raw2raw) | raw-to-raw | [unsupervised](configs/) | [checkpoint](https://github.com/gosha20777/cmKAN/releases) |
| [Zurich raw-to-sRGB](https://people.ee.ethz.ch/~ihnatova/pynet.html) | raw-to-sRGB | [supervised](configs/) | [checkpoint](https://github.com/gosha20777/cmKAN/releases) |

## Training on a Custom Unpaired Dataset

Put the two independent image domains under one dataset root. Source and target
do not need matching filenames or the same number of images:

```text
data/my_dataset/
├── train/
│   ├── source/
│   │   ├── image_001.jpg
│   │   └── ...
│   └── target/
│       ├── another_name.png
│       └── ...
└── val/
    ├── source/
    └── target/
```

Run the provided training script:

```bash
./scripts/train_custom_unpaired.sh data/my_dataset
```

A complete Chinese training, inference, and troubleshooting guide is available at
[`docs/custom_unpaired_training_zh.md`](docs/custom_unpaired_training_zh.md).

Domain directory names can be supplied when they are not `source` and `target`.
For example, a Samsung-to-iPhone dataset can be launched with:

```bash
./scripts/train_custom_unpaired.sh data/my_dataset samsung iphone
```

If a train domain contains a `real/` subdirectory, the script selects it and
does not mix in a sibling `recolor/` directory. An existing validation split is
used directly; only datasets without `val/` fall back to automatic splitting.
If `test/` is absent, test-time evaluation reuses `val/`.

After training, run loss evaluation and generate full-resolution predictions in
both directions with one command:

```bash
CUDA_VISIBLE_DEVICES=7 ./scripts/test_custom_unpaired.sh
```

The script passes `--reverse` as a flag (without a trailing `1`), writes test
metrics separately from the training CSV, and saves both translation directions
under `results/custom_unpaired/` by default.

For privacy-safe debugging without sharing images or filenames, print aggregate
brightness, contrast, clipping, and RGB statistics:

```bash
python scripts/diagnose_prediction_stats.py
```

The scripts default to `/home/share/y50063074/data` and
`results/custom_unpaired`. Override them with `CMKAN_DATA_ROOT` and
`CMKAN_RESULTS_ROOT` or the existing command-line arguments.

The loader recursively discovers common image formats. Training uses resize,
random crop, and horizontal/vertical flip augmentation. Color-changing
augmentation is omitted because it would alter the source and target color
distributions. The custom configuration supports a non-adversarial generator
warm-up, gradient clipping, output-range regularization, differentiable
color/exposure moments,
intensity-invariant chromaticity consistency, and log-domain reflectance
consistency. The last two terms keep subject color and local intrinsic contrast
while still permitting smooth illumination changes. Optional scene-grouped sampling
restricts source/target matching to corresponding relative subdirectories.
`pairing_mode: weak_aligned` additionally creates a fixed rough correspondence,
including unequal domain sizes. `pairing_mode: one_to_one` requires a bijection:
each source uses exactly one target and no target is repeated or dropped. Both
aligned modes synchronize geometric augmentation while still avoiding pixel losses
on imperfectly aligned pairs. PatchNCE compares cmKAN contextual patches between
each input and its own translation to preserve content.
Adjust image size, split ratios, batch size, and training length in
`configs/custom_unpaired.example.yaml`.

### Reference-guided unpaired color transfer

When the target domain contains several color temperatures or exposure styles,
use the reference-guided model so each translation follows one selected target
image instead of collapsing to an average target look. The dataset layout stays
the same. The supplied reference configs use strict one-to-one sampling: a complete
relative-filename correspondence is used when available; otherwise each matching
scene directory is naturally sorted and zipped. Counts must match globally and in
every scene directory, so a target can never be silently reused.

Reference-guided checkpoints contain additional conditioning layers. For the
current skin-aware run, preview the private masks and then train from scratch:

```bash
python scripts/preview_skin_masks.py

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 \
./scripts/train_custom_unpaired_reference_v5_skin.sh
```

The launcher stops early with an installation hint if it detects the known
Lightning 2.1.x and Rich 14+ progress-bar incompatibility.

The server config starts a separate experiment named
`custom_one_to_one_reference_color_v5_skin`, so existing v1/v2/v3/v4/v6
checkpoints and CSV logs are left untouched. v4 addressed a measured
`ratio=0.9972` with `luma_ratio=1.03`: exposure was stable, but the output was
almost identical to the source and the reference branch was effectively
ignored. It scales the style delta by 10, applies a smooth `tanh` bound, and
adds a zero-initialized direct path from that condition to the spatial KAN
parameter tensor. The contextual branch still supplies spatial variation.
Both conditional output heads start at zero, so the zero-update output remains
the source image while useful condition gradients are available on the first
update.

v5 keeps that conditioning path, restores the safe global style weight to `3`,
and adds a non-aligned skin-tone objective. The generated image is measured
through a mask computed from the real source; reference statistics use a
separate mask from the real target. Linear-RGB log-chroma moments and a small
luminance term are compared without matching pixels or copying facial
structure. Skin-local uniformity and target-relative red guards protect against
red patches and uniformly over-red faces.

The v5 stability configuration trains for 200 epochs. It keeps the bounded
output and local color safeguards, uses five epochs of the complete
non-adversarial generator objective, then ramps the adversarial weight from
`0.1` at epoch 5 to `1.0` at epoch 14. Keep `resume: false`. Do not resume the
finished style-15 red checkpoint: its optimizer schedule is already decayed and
its learned color bias would be retained.

Before training updates, the callback saves
`experiments/custom_one_to_one_reference_color_v5_skin/logs/figures/initial_source_to_target_0.png`;
it should look like the source because the bounded residual head starts from
identity. The sibling `source_to_target_0.png` is saved after the first complete
warm-up epoch. Reference-guided best checkpoints monitor
`val_reference_selection_loss`, which excludes the changing discriminator
score and includes the target-relative skin objective. Run
`python scripts/report_reference_metrics.py --skin` for the skin report, or
`python scripts/report_reference_metrics.py` for the compact
`ratio/move/response/direct/luma_ratio/red_bad` report. During epochs 1–5,
`response` and `direct` should leave zero and `skin_valid` must be non-zero.
The color mask is heuristic, so inspect `skin_mask_preview.png`: white regions
should cover skin instead of wood, yellow walls, or clothing. The v2/v3/v4
configs and launchers remain available for exact rollback. Samples below 0.5%
or above 50% candidate coverage are excluded from the skin objective.

Use one target image as the reference for one source image or a source folder:

```bash
CUDA_VISIBLE_DEVICES=7 python main.py predict \
  --config configs/custom_unpaired_reference_v5_skin.server.yaml \
  --weights logs/checkpoints/last.ckpt \
  --input /absolute/path/to/source_images \
  --reference /absolute/path/to/target_reference.jpg \
  --output results/reference_guided \
  --batch_size 1
```

The reference contributes global linear-RGB, CIE chromaticity, luminance, and
contrast statistics; it does not copy the reference scene or subject. See the
[Chinese custom unpaired guide](docs/custom_unpaired_training_zh.md#-参考图引导模式当前数据推荐)
for the training loss, configuration rationale, and a same-source/different-reference
comparison procedure. Prediction must use the same v5 YAML as training because
the architecture and bounded-output parameters are selected by configuration.



## How To Use

Our `cmKAN` provides a command-line interface (CLI) to interact with the following tools:

```bash
python main.py -h 

Usage: cmKAN CLI [-h] {data-create,test,train,predict,unit-test} ...

Options:
  -h, --help            Show this help message and exit

Tools:
  {data-create,test,train,predict,unit-test}
    
    data-create         Create dataset
    train               Train model
    test                Test model
    predict             Run model inference
    unit-test           Run unit tests
```

For all the tools, you can use the `-h` flag to get help on how to use them (e.g. `python main.py train -h`). Here are some examples on how to use the tools:

### Train

```bash
python main.py train -c configs/config.yaml
```

### Test

```bash
python main.py test -c configs/config.yaml -w checkpoint.ckpt
```

### Predict

```bash
python main.py predict -c configs/config.yaml -i path/to/input/folder -o path/to/output/folder
```

### Additional Guides

You can find additional guides how to reproduce all our experiments in [our wiki page](https://github.com/gosha20777/cmKAN/wiki). We provide detailed instructions on how to train and test our model, as well as how to use it for inference. 

## Citation

If you find our work useful, please consider citing it:

```bibtex
@article{perevozchikov2025color,
  title={Color Matching Using Hypernetwork-Based Kolmogorov-Arnold Networks},
  author={Nikonorov, Artem and Perevozchikov, Georgy and Korepanov, Andrei and Mehta, Nancy and Afifi, Mahmoud and Ershov, Egor and Timofte, Radu},
  journal={arXiv preprint arXiv:2503.11781},
  year={2025}
}
```
