#!/usr/bin/env python3
"""
Horizyn Model Training Script

Trains the Horizyn contrastive learning model for enzyme-reaction matching.

Usage:
    python train.py --config configs/data1_t5_full_pocket_msa_projected_gate.yaml

    # Override config values
    python train.py --config configs/data1_t5_full_pocket_msa_projected_gate.yaml --training.max_epochs 50

    # Set random seed
    python train.py --config configs/data1_t5_full_pocket_msa_projected_gate.yaml --seed 123

Requirements:
    - Prepared pair tables and protein embeddings are required
    - Requires ~16GB RAM (all data loaded into memory)
    - Requires single GPU with 16GB+ VRAM

Example:
    # Train the combined evolutionary variant
    python train.py --config configs/data1_t5_full_pocket_msa_projected_gate.yaml

    # Train with custom batch size
    python train.py --config configs/data1_t5_full_pocket_msa_projected_gate.yaml --data.train_batch_size 8192
"""

import argparse
import sys
from pathlib import Path

import lightning.pytorch as pl
import torch

from horizyn.config import load_config, parse_overrides
from horizyn.data_module import HorizynDataModule
from horizyn.lightning_module import HorizynLitModule


def load_initial_weights(
    model: HorizynLitModule,
    checkpoint_path: str,
    target_encoder_name: str,
) -> None:
    """Load compatible weights from a checkpoint without restoring trainer state."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source_state = checkpoint.get("state_dict", checkpoint)
    target_state = model.state_dict()
    mapped_state = {}

    for key, value in source_state.items():
        candidate_keys = [key]

        if key.startswith("model.query_encoder.main_nn."):
            candidate_keys.append(
                key.replace(
                    "model.query_encoder.main_nn.",
                    "model.query_encoder.base_encoder.main_nn.",
                    1,
                )
            )
        elif key.startswith("model.query_encoder.post_nn_layers."):
            candidate_keys.append(
                key.replace(
                    "model.query_encoder.post_nn_layers.",
                    "model.query_encoder.base_encoder.post_nn_layers.",
                    1,
                )
            )

        if key.startswith("model.target_encoder.main_nn."):
            candidate_keys.append(
                key.replace(
                    "model.target_encoder.main_nn.",
                    "model.target_encoder.encoder.main_nn.",
                    1,
                )
            )
            candidate_keys.append(
                key.replace(
                    "model.target_encoder.main_nn.",
                    "model.target_encoder.t5_encoder.main_nn.",
                    1,
                )
            )
        elif key.startswith("model.target_encoder.post_nn_layers."):
            candidate_keys.append(
                key.replace(
                    "model.target_encoder.post_nn_layers.",
                    "model.target_encoder.encoder.post_nn_layers.",
                    1,
                )
            )
            candidate_keys.append(
                key.replace(
                    "model.target_encoder.post_nn_layers.",
                    "model.target_encoder.t5_encoder.post_nn_layers.",
                    1,
                )
            )

        for candidate_key in candidate_keys:
            if candidate_key in target_state and target_state[candidate_key].shape == value.shape:
                mapped_state[candidate_key] = value
                break

    missing, unexpected = model.load_state_dict(mapped_state, strict=False)
    print(f"\nInitialized from checkpoint: {checkpoint_path}")
    print(f"  Loaded tensors: {len(mapped_state)}")
    print(f"  Missing tensors after init: {len(missing)}")
    print(f"  Unexpected tensors after init: {len(unexpected)}\n")


def freeze_non_msa_adapter_parameters(model: HorizynLitModule) -> None:
    """Freeze everything except the MSA adapter in late-fusion target encoders."""
    trainable_name_parts = (
        "model.query_encoder.aux_encoder",
        "model.query_encoder.aux_gate",
        "model.target_encoder.msa_norm",
        "model.target_encoder.full_msa_norm",
        "model.target_encoder.pocket_msa_norm",
        "model.target_encoder.msa_projection",
        "model.target_encoder.full_msa_projection",
        "model.target_encoder.pocket_msa_projection",
        "model.target_encoder.projected_gate_norm",
        "model.target_encoder.full_projected_gate_norm",
        "model.target_encoder.pocket_projected_gate_norm",
        "model.target_encoder.msa_encoder",
        "model.target_encoder.gate",
        "model.target_encoder.full_gate",
        "model.target_encoder.pocket_gate",
    )
    trainable = 0
    frozen = 0
    for name, param in model.named_parameters():
        should_train = any(part in name for part in trainable_name_parts)
        param.requires_grad = should_train
        if should_train:
            trainable += param.numel()
        else:
            frozen += param.numel()

    print("\nFreezing non-MSA-adapter parameters")
    print(f"  Trainable adapter parameters: {trainable:,}")
    print(f"  Frozen parameters: {frozen:,}\n")


def main():
    """Main training function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train Horizyn contrastive learning model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to an evolutionary-augmentation YAML config",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (overrides config.seed if provided)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--init-from",
        type=str,
        default=None,
        help=(
            "Path to a checkpoint used only for non-strict weight initialization. "
            "Unlike --resume, optimizer/trainer state is not restored."
        ),
    )

    # Parse known args and capture remaining for overrides
    args, unknown = parser.parse_known_args()

    # Parse config overrides from remaining arguments
    overrides = parse_overrides(unknown)

    # Apply seed override if provided
    if args.seed is not None:
        overrides["seed"] = args.seed

    # Load configuration
    print(f"Loading config from: {args.config}")
    try:
        config = load_config(args.config, overrides=overrides)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure you're running from the project root directory.")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: Config validation failed")
        print(f"{e}")
        sys.exit(1)

    # Print configuration summary
    print("\n" + "=" * 80)
    print("HORIZYN TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"Seed: {config.seed}")
    print(f"Max Epochs: {config.training.max_epochs}")
    print(f"Train Batch Size: {config.data.train_batch_size}")
    print(f"Retrieval Batch Size: {config.data.retrieval_batch_size}")
    print(f"Learning Rate: {config.training.learning_rate}")
    print(f"Weight Decay: {config.training.weight_decay}")
    print(f"Model: {config.model.name}")
    print(f"Query Encoder: {config.model.query_encoder_dims}")
    print(f"Target Encoder: {config.model.target_encoder_dims}")
    print(f"Target Encoder Type: {config.model.get('target_encoder_name', 'MLP')}")
    print(f"Embedding Dim: {config.model.embedding_dim}")
    print(f"Loss: {config.training.loss.name} (beta={config.training.loss.beta})")
    print(f"Log Dir: {config.logging.log_dir}")
    print(f"Checkpoint Dir: {config.logging.checkpoint_dir}")
    print("=" * 80 + "\n")

    # Set random seed for reproducibility
    seed = config.get("seed", 42)
    pl.seed_everything(seed, workers=True)
    print(f"Set random seed to: {seed}\n")

    # Check for GPU availability
    if not torch.cuda.is_available():
        print("Warning: No GPU detected. Training will be very slow on CPU.")
        print("Consider using a machine with a CUDA-capable GPU.\n")

    # Setup data module
    print("Initializing data module...")
    try:
        data_module = HorizynDataModule(
            train_pairs_path=config.data.train_pairs_path,
            test_pairs_path=config.data.test_pairs_path,
            train_reactions_path=config.data.train_reactions_path,
            test_reactions_path=config.data.test_reactions_path,
            protein_embeds_path=config.data.protein_embeds_path,
            train_batch_size=config.data.train_batch_size,
            retrieval_batch_size=config.data.retrieval_batch_size,
            rdkit_fp_dim=config.data.get("rdkit_fp_dim", 1024),
            drfp_dim=config.data.get("drfp_dim", 1024),
            num_workers=config.data.get("num_workers", 0),
            pin_memory=config.data.get("pin_memory", False),
            standardize_reactions=config.data.get("standardize_reactions", True),
            standardize_hypervalent=config.data.get("standardize_hypervalent", True),
            standardize_remove_hs=config.data.get("standardize_remove_hs", True),
            standardize_kekulize=config.data.get("standardize_kekulize", False),
            standardize_uncharge=config.data.get("standardize_uncharge", True),
            standardize_metals=config.data.get("standardize_metals", True),
            standardize_error_policy=config.data.get("standardize_error_policy", "raise"),
            fingerprint_error_policy=config.data.get("fingerprint_error_policy", "raise"),
        )
    except FileNotFoundError as e:
        print(f"\nError: Data file not found")
        print(f"{e}")
        print("\nPlease download the dataset first:")
        print("    See README.md for data preparation.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError initializing data module: {e}")
        sys.exit(1)

    print("Data module initialized.\n")

    # Setup model
    print("Initializing model...")
    model = HorizynLitModule(
        query_encoder_dims=config.model.query_encoder_dims,
        target_encoder_dims=config.model.target_encoder_dims,
        embedding_dim=config.model.embedding_dim,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        beta=config.training.loss.beta,
        learn_beta=config.training.loss.get("learn_beta", False),
        metric_ks=config.training.metrics.get("top_k", [1, 10, 100, 1000]),
        msa_adapter_lr_multiplier=config.training.get("msa_adapter_lr_multiplier", 1.0),
        query_encoder_name=config.model.get("query_encoder_name", "MLP"),
        query_aux_encoder_widths=config.model.get("query_aux_encoder_widths", 2048),
        view_aux_weight=config.model.get("view_aux_weight", 0.35),
        query_dynamic_aux_weight=config.model.get("query_dynamic_aux_weight", False),
        query_aux_weight_min=config.model.get("query_aux_weight_min", 0.05),
        query_aux_weight_max=config.model.get("query_aux_weight_max", 0.75),
        target_encoder_name=config.model.get("target_encoder_name", "MLP"),
        target_t5_dim=config.model.get("target_t5_dim", 1024),
        target_msa_dim=config.model.get("target_msa_dim", 768),
        target_fusion_dim=config.model.get("target_fusion_dim", 1024),
        target_msa_encoder_widths=config.model.get("target_msa_encoder_widths", 1024),
        target_msa_gate_bias_init=config.model.get("target_msa_gate_bias_init", -4.0),
        target_msa_fusion_mode=config.model.get("target_msa_fusion_mode", "residual"),
        target_msa_projection_hidden_dim=config.model.get(
            "target_msa_projection_hidden_dim", 2048
        ),
        target_msa_gate_hidden_dim=config.model.get("target_msa_gate_hidden_dim", 1024),
        target_msa_gate_type=config.model.get("target_msa_gate_type", "vector"),
        target_msa_alignment_loss_weight=config.model.get(
            "target_msa_alignment_loss_weight", 0.05
        ),
        target_msa_gate_l1_weight=config.model.get("target_msa_gate_l1_weight", 0.001),
        target_msa_projection_dropout=config.model.get("target_msa_projection_dropout", 0.1),
        dualview_aux_loss_weight=config.training.get("dualview_aux_loss_weight", 0.0),
    )

    if args.init_from is not None:
        load_initial_weights(
            model,
            args.init_from,
            target_encoder_name=config.model.get("target_encoder_name", "MLP"),
        )

    if config.training.get("freeze_non_msa_adapter", False):
        freeze_non_msa_adapter_parameters(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")

    # Setup logging
    logger = pl.loggers.CSVLogger(
        save_dir=config.logging.log_dir,
        name="horizyn_training",
    )

    # Setup callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=config.logging.checkpoint_dir,
        filename="horizyn-{epoch:02d}",
        every_n_epochs=config.logging.get("save_every_n_epochs", 10),
        save_last=True,
        save_top_k=3,
        monitor="val/loss",
        mode="min",
    )

    # Setup trainer
    print("Setting up Lightning Trainer...")
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=config.logging.get("log_every_n_steps", 1),
        check_val_every_n_epoch=config.training.get("check_val_every_n_epoch", 10),
        enable_progress_bar=config.training.get("enable_progress_bar", True),
        deterministic=True,  # For reproducibility
        # Single GPU training (DDP not supported in simplified version)
        devices=1 if torch.cuda.is_available() else "auto",
        accelerator="auto",
    )

    print(f"Trainer configured for {config.training.max_epochs} epochs\n")

    # Train
    print("=" * 80)
    print("STARTING TRAINING")
    print("=" * 80 + "\n")

    try:
        trainer.fit(
            model,
            datamodule=data_module,
            ckpt_path=args.resume,  # Resume from checkpoint if provided
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print(f"Last checkpoint saved to: {checkpoint_callback.last_model_path}")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Training complete
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Last checkpoint: {checkpoint_callback.last_model_path}")
    print(f"Logs saved to: {config.logging.log_dir}")
    print("\nTo resume training, use:")
    print(
        f"    python train.py --config {args.config} --resume {checkpoint_callback.last_model_path}"
    )


if __name__ == "__main__":
    main()
