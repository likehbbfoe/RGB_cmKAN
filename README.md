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

Reference-guided checkpoints contain additional conditioning layers. When faces
appear against beige, yellow, wood, or other skin-like backgrounds, use the v6
face-ROI run. It intersects the v5 color candidate with a precomputed face mask,
so background color cannot occupy most of the skin objective.

Generate the private sidecar masks on the training server:

```bash
python scripts/generate_face_masks.py
```

Inspect `/home/share/y50063074/data_face_masks/face_mask_preview.png` before
training. Its third column is the actual `face ROI × skin color` mask used by
the loss; white must cover facial skin rather than the surrounding background.
The mask tree mirrors the dataset tree, and every image must have a `.png`
sidecar:

```text
/home/share/y50063074/data_face_masks/
├── train/
│   ├── source/
│   └── target/
└── val/
    ├── source/
    └── target/
```

A missing sidecar stops immediately instead of silently falling back to the v5
color-only heuristic. A present but completely black mask means no reliable
face was detected; that sample still trains with the other objectives, but
contributes nothing to the skin loss.

After checking the preview, start a fresh v6 experiment:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 \
./scripts/train_custom_unpaired_reference_v6_face_skin.sh
```

The launcher stops early with an installation hint if it detects the known
Lightning 2.1.x and Rich 14+ progress-bar incompatibility. The server config
starts a separate experiment named
`custom_one_to_one_reference_color_v6_face_skin`, so all v1–v5 checkpoints and
CSV logs are left untouched.

v4 scales the reference style delta by 10 and adds a zero-initialized direct path
from that condition to the spatial KAN parameter tensor. v5 keeps that path,
restores the safe global style weight to `3`, and adds target-relative skin-tone
statistics and local red guards. v6 keeps those weights and protections, but
restricts the source and target color masks to their corresponding face ROIs.
It still compares color statistics rather than facial pixels, so it does not
copy the target person's features or texture. Conservative validity gates also
skip ROIs that are too small/large, nearly uniform skin color, or more than 2×
different in source/target area, as well as pairs whose face centers are far
apart.

The v6 stability configuration trains for 200 epochs. It uses five epochs of
the complete non-adversarial generator objective, then ramps the adversarial
weight from `0.1` at epoch 5 to `1.0` at epoch 14. Keep `resume: false`; do not
resume a v5 or older checkpoint. Geometric augmentation applies the same resize,
crop, and flip to each image and its sidecar mask.

Before training updates, the callback saves
`experiments/custom_one_to_one_reference_color_v6_face_skin/logs/figures/initial_source_to_target_0.png`;
it should look like the source because the bounded residual head starts from
identity. Reference-guided best checkpoints monitor
`val_reference_selection_loss`. Run
`python scripts/report_reference_metrics.py --skin` for the skin report. During
epochs 1–5, `response` and `direct` should leave zero and `skin_valid` must be
non-zero. A black face mask only lowers the number of valid skin samples; it
does not disable adversarial, cycle, identity, exposure, reflectance, or PatchNCE
training. The v1–v5 configs remain available for exact rollback.

Use one target image as the reference for one source image or a source folder:

```bash
CUDA_VISIBLE_DEVICES=7 python main.py predict \
  --config configs/custom_unpaired_reference_v6_face_skin.server.yaml \
  --weights logs/checkpoints/last.ckpt \
  --input /absolute/path/to/source_images \
  --reference /absolute/path/to/target_reference.jpg \
  --output results/reference_guided \
  --batch_size 1
```

The reference contributes global color and exposure statistics; it does not
copy the reference scene or subject. See the
[Chinese custom unpaired guide](docs/custom_unpaired_training_zh.md#-参考图引导模式当前数据推荐)
for the complete mask checks and training procedure. Prediction must use the
same v6 YAML as training.



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
