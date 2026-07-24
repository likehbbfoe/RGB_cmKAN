import argparse
import yaml
from ..core import Logger
from ..core.selector import (
    ModelSelector,
    PipelineSelector
)
from ..core.config import Config
from ..core.config.model import ModelType
from ..core.config.pipeline import PipelineType
from ..ml.datasets import ImgPredictDataModule
import lightning as L
import os
from lightning.pytorch.callbacks import (
    RichModelSummary,
    RichProgressBar,
)
from cm_kan.ml.callbacks import ImagePredictionWriter
from lightning.pytorch.loggers import CSVLogger
from cm_kan import cli
from .experiment_paths import (
    experiment_directory,
    prediction_output_directory,
)


class _ReferencePathAction(argparse.Action):
    """Store a reference path while remembering that the CLI flag was explicit."""

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        setattr(namespace, self.dest, values)
        setattr(namespace, "reference_provided", True)


def add_parser(subparser: argparse) -> None:
    parser = subparser.add_parser(
        "predict",
        help="Process images with a trained model",
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
        "-w", "--weights",
        type=str,
        help="Path to checkpoint file in the experiment folder",
        default="logs/checkpoints/last.ckpt",
        required=False,
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        help="Path to one input image or an input image folder",
        default="data/samples/input",
        required=False,
    )
    parser.add_argument(
        "-r", "--reference",
        type=str,
        help=(
            "Reference image or folder. Reference-guided models accept one image "
            "for all inputs, a one-image folder, or a folder matching the input count"
        ),
        default="data/samples/reference",
        action=_ReferencePathAction,
        required=False,
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help=(
            "Path to the output folder. Defaults to "
            "<save_dir>/<experiment>/predictions"
        ),
        default=None,
        required=False,
    )
    parser.add_argument(
        "-bs", "--batch_size",
        type=int,
        help="Batch size for prediction",
        default=1,
        required=False,
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Reverse the direction of color transfer (for unpaired scenario only)",
        required=False,
    )

    parser.set_defaults(func=predict, reference_provided=False)


def predict(args: argparse.Namespace) -> None:
    if not (os.path.isfile(args.input) or os.path.isdir(args.input)):
        raise ValueError(
            f"Incorrect input path '{args.input}'. It should be an image file "
            "or directory."
        )

    Logger.info(f"Loading config from '{args.config}'")
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    config = Config(**config)
    experiment_dir = experiment_directory(
        config.save_dir,
        config.experiment,
    )
    output_dir = prediction_output_directory(
        config.save_dir,
        config.experiment,
        args.output,
    )

    reference_guided = config.model.type == ModelType.reference_cycle_cm_kan

    if reference_guided:
        reference_provided = getattr(
            args,
            "reference_provided",
            args.reference is not None,
        )
        if not reference_provided:
            raise ValueError(
                "--reference is required for reference-guided prediction. "
                "Pass one target-style image or a reference directory."
            )
        if not (os.path.isfile(args.reference) or os.path.isdir(args.reference)):
            raise ValueError(
                f"Incorrect reference path '{args.reference}'. It should be an "
                "image file or directory."
            )
    elif config.pipeline.type == PipelineType.pair_based:
        if args.reference is None or not os.path.isdir(args.reference):
            raise ValueError(
                f"Incorrect reference path '{args.reference}'. It should be a directory."
            )

    inference_mode = config.pipeline.type != PipelineType.pair_based
    if not inference_mode:
        Logger.info(f'Inference mode: {inference_mode}. Use optimization while testing.')
    Logger.info('Config:')
    config.print()
    Logger.info(f"Experiment directory: '{experiment_dir}'")
    Logger.info(f"Prediction output directory: '{output_dir}'")
    
    dm = ImgPredictDataModule(
        input_path=args.input,
        reference_path=args.reference,
        pipeline_type=config.pipeline.type,
        reference_guided=reference_guided,
        batch_size=args.batch_size,
    )
    model = ModelSelector.select(config)
    pipeline = PipelineSelector.select(config, model, reverse_prediction=args.reverse)

    logger = CSVLogger(
        save_dir=experiment_dir,
        name='predict_logs',
        version='',
    )

    trainer = L.Trainer(
        logger=logger,
        default_root_dir=experiment_dir,
        max_epochs=config.pipeline.params.epochs,
        accelerator=config.accelerator,
        callbacks=[
            RichModelSummary(),
            RichProgressBar(),
            ImagePredictionWriter(
                output_dir=output_dir,
                write_interval='batch',
            ),
        ],
        inference_mode=inference_mode,
    )

    ckpt_path = os.path.join(experiment_dir, args.weights)

    if not os.path.exists(ckpt_path):
        raise ValueError(f"Checkpoint file '{ckpt_path}' does not exist.")

    trainer.predict(
        model=pipeline, 
        datamodule=dm,
        ckpt_path=ckpt_path,
        return_predictions=False
    )
