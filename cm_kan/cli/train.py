import argparse
import yaml
from ..core import Logger
from ..core.selector import (
    ModelSelector,
    DataSelector,
    PipelineSelector
)
from ..core.config import Config
from ..core.config.data import DataType
import lightning as L
import os
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
    LearningRateMonitor,
)
from cm_kan.ml.callbacks import GenerateCallback
from lightning.pytorch.loggers import CSVLogger
from cm_kan import cli


def _domain_path(data_root: str, split: str, domain: str) -> str:
    split_root = os.path.join(data_root, split)
    path = (
        os.path.join(split_root, domain)
        if os.path.isdir(split_root)
        else os.path.join(data_root, domain)
    )
    real_path = os.path.join(path, "real")
    if split == "train" and os.path.isdir(real_path):
        return real_path
    return path


def add_parser(subparser: argparse) -> None:
    parser = subparser.add_parser(
        "train",
        help="Train color transfer model",
        formatter_class=cli.ArgumentDefaultsRichHelpFormatter,
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        help="Path to config file",
        default="config.yaml",
        required=False,
    )
    parser.add_argument(
        "--data-root",
        type=str,
        help="Override custom dataset root containing source/ and target/",
        default=None,
    )
    parser.add_argument(
        "--source-domain",
        type=str,
        help="Source-domain directory name below train/ and val/",
        default="source",
    )
    parser.add_argument(
        "--target-domain",
        type=str,
        help="Target-domain directory name below train/ and val/",
        default="target",
    )

    parser.set_defaults(func=train)


def train(args: argparse.Namespace) -> None:
    Logger.info(f"Loading config from '{args.config}'")
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if args.data_root is not None:
        if config.get("data", {}).get("type") != DataType.custom_unpaired.value:
            raise ValueError("--data-root can only be used with data.type=custom_unpaired")
        config["data"]["train"] = {
            "source": _domain_path(args.data_root, "train", args.source_domain),
            "target": _domain_path(args.data_root, "train", args.target_domain),
        }
        if os.path.isdir(os.path.join(args.data_root, "val")):
            config["data"]["val"] = {
                "source": _domain_path(args.data_root, "val", args.source_domain),
                "target": _domain_path(args.data_root, "val", args.target_domain),
            }
        else:
            config["data"].pop("val", None)

        if os.path.isdir(os.path.join(args.data_root, "test")):
            config["data"]["test"] = {
                "source": _domain_path(args.data_root, "test", args.source_domain),
                "target": _domain_path(args.data_root, "test", args.target_domain),
            }
        else:
            config["data"].pop("test", None)

    config = Config(**config)
    if config.data.type == DataType.custom_unpaired:
        L.seed_everything(config.data.params.seed, workers=True)
    Logger.info('Config:')
    config.print()
    
    dm = DataSelector.select(config)
    model = ModelSelector.select(config)
    pipeline = PipelineSelector.select(config, model)

    logger = CSVLogger(
        save_dir=os.path.join(config.save_dir, config.experiment),
        name='logs',
        version='',
    )

    is_custom_unpaired = config.data.type == DataType.custom_unpaired
    checkpoint_monitor = 'val_loss' if is_custom_unpaired else 'val_de'
    checkpoint_filename = (
        "{epoch}-{val_loss:.4f}"
        if is_custom_unpaired
        else "{epoch}-{val_de:.2f}"
    )

    trainer = L.Trainer(
        logger=logger,
        default_root_dir=os.path.join(config.save_dir, config.experiment),
        max_epochs=config.pipeline.params.epochs,
        accelerator=config.accelerator,
        callbacks=[
            ModelCheckpoint(
                filename=checkpoint_filename,
                monitor=checkpoint_monitor,
                save_last=True,
                every_n_epochs=config.pipeline.params.save_freq,
            ),
            RichModelSummary(),
            RichProgressBar(),
            LearningRateMonitor(
                logging_interval='epoch',
            ),
            GenerateCallback(
                every_n_epochs=config.pipeline.params.visualize_freq,
            ),
        ],
    )

    ckpt_path = os.path.join(config.save_dir, config.experiment, 'logs/checkpoints/last.ckpt')

    trainer.fit(
        model=pipeline, 
        datamodule=dm,
        ckpt_path=ckpt_path if config.resume and os.path.exists(ckpt_path) else None,
    )
