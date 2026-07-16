"""
Neural network model architectures for Horizyn.

This module contains the base model classes and MLP implementation used in
the Horizyn contrastive learning model.
"""

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    """
    Base class for all models in Horizyn.

    Provides a structured way to organize model layers into pre-processing,
    main body, and post-processing stages, along with optional output heads.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the base model.

        Args:
            *args: Variable length argument list (should be empty).
            **kwargs: Arbitrary keyword arguments (should be empty).

        Raises:
            ValueError: If extra arguments are provided.
        """
        super(BaseModel, self).__init__()
        if args or kwargs:
            error_msg = (
                f"Extra unused arguments provided to BaseModel: args={args}, kwargs={kwargs}"
            )
            raise ValueError(error_msg)

        # Define the pre-nn layers (preprocessing)
        self.pre_nn_layers = nn.ModuleList()
        # Define the main body of nn layers
        self.main_nn = nn.ModuleList()
        # Define the post-nn layers (postprocessing)
        self.post_nn_layers = nn.ModuleList()
        # Optional output heads for multi-task learning
        self.output_heads = nn.ModuleDict()

    @property
    def model_body(self) -> nn.ModuleList:
        """
        Get the main body of the model (all layers excluding output heads).

        Returns:
            ModuleList containing all pre-processing, main, and post-processing layers.
        """
        return nn.ModuleList([*self.pre_nn_layers, *self.main_nn, *self.post_nn_layers])

    @property
    def layers(self) -> nn.ModuleList:
        """
        Get all layers in the model including output heads.

        Returns:
            ModuleList containing all model layers.
        """
        return nn.ModuleList(
            [
                *self.pre_nn_layers,
                *self.main_nn,
                *self.post_nn_layers,
                *self.output_heads.values(),
            ]
        )

    @property
    def num_parameters(self) -> int:
        """
        Get the total number of parameters in the model.

        Returns:
            Total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            x: Input tensor.

        Returns:
            Output tensor, or dict of outputs if output heads are defined.
        """
        for layer in self.model_body:
            x = layer(x)
        # Handle multiple output heads if present
        if len(self.output_heads) > 0:
            return {key: head(x) for key, head in self.output_heads.items()}
        return x


class NormalizeLayer(nn.Module):
    """
    Normalization layer for L2 normalization of tensors.

    This layer normalizes input tensors along a specified dimension using the
    L2 norm (Euclidean distance). Commonly used to normalize embeddings in
    contrastive learning.
    """

    def __init__(self, p: float = 2, dim: int = -1, eps: float = 1e-12):
        """
        Initialize the NormalizeLayer.

        Args:
            p: The p-norm to use for normalization (default: 2 for L2 norm).
            dim: The dimension along which to compute the norm (default: -1, last dimension).
            eps: Small value for numerical stability (default: 1e-12).
        """
        super(NormalizeLayer, self).__init__()
        self.p = p
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply normalization to the input tensor.

        Args:
            x: Input tensor to be normalized.

        Returns:
            Normalized tensor with unit norm along the specified dimension.
        """
        return F.normalize(x, p=self.p, dim=self.dim, eps=self.eps)

    def extra_repr(self) -> str:
        """
        Return string representation of layer parameters for printing.

        Returns:
            String describing layer configuration.
        """
        return f"p={self.p}, dim={self.dim}, eps={self.eps}"


class MLP(BaseModel):
    """
    Multi-Layer Perceptron (MLP) neural network.

    Implements a flexible MLP architecture with customizable layers, activation
    functions, layer normalization, dropout, and optional output normalization.
    This is the primary encoder architecture used in the Horizyn SOTA model.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: int = 1,
        widths: int | list[int] = 32,
        activations: nn.Module | list[nn.Module] = nn.ReLU(),
        use_layer_norm: bool = False,
        dropout: float = 0.0,
        bias: bool = True,
        normalise_output: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize the MLP.

        Args:
            input_dim: Dimension of the input features.
            output_dim: Dimension of the output features.
            num_layers: Number of hidden layers (default: 1).
            widths: Width(s) of hidden layers. If int, all layers have same width.
                If list, each element specifies width of corresponding layer.
            activations: Activation function(s). If single Module, used for all layers.
                If list, each element specifies activation for corresponding layer.
            use_layer_norm: Whether to apply layer normalization after each hidden layer.
            dropout: Dropout probability (0.0 means no dropout).
            bias: Whether to include bias terms in linear layers.
            normalise_output: Whether to L2-normalize the final output.
            *args: Additional arguments (must be empty).
            **kwargs: Additional keyword arguments (must be empty).

        Example:
            >>> # SOTA reaction encoder: 2048 → 4096 → 512
            >>> mlp = MLP(
            ...     input_dim=2048,
            ...     output_dim=512,
            ...     num_layers=1,
            ...     widths=4096,
            ...     normalise_output=True
            ... )
        """
        super(MLP, self).__init__(*args, **kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Validate core hyperparameters early (fail fast)
        if num_layers < 0:
            raise ValueError("num_layers must be >= 0")
        if isinstance(widths, int):
            if widths <= 0 and num_layers > 0:
                raise ValueError("widths must be a positive integer when num_layers > 0")
        else:
            if len(widths) == 0 and num_layers > 0:
                raise ValueError("widths list must be non-empty when num_layers > 0")
            if any(w <= 0 for w in widths):
                raise ValueError("all hidden layer widths must be positive integers")
        if not (0.0 <= dropout <= 1.0):
            raise ValueError("dropout must be in the range [0.0, 1.0]")

        # Build the main neural network
        self._build_network(num_layers, widths, activations, use_layer_norm, dropout, bias)

        # Add output normalization if requested
        if normalise_output:
            self.post_nn_layers.append(NormalizeLayer(p=2, dim=-1))

    def _build_network(
        self,
        num_layers: int,
        widths: int | list[int],
        activations: nn.Module | list[nn.Module],
        use_layer_norm: bool,
        dropout: float,
        bias: bool,
    ) -> None:
        """
        Build the main neural network structure.

        Constructs the layers of the MLP based on the provided parameters,
        including linear layers, activations, layer normalization, and dropout.

        Notes:
            - If `widths` is a list, its length defines the number of hidden layers
              and overrides `num_layers`.
            - If `activations` is provided as a single nn.Module instance, a deep copy
              of that instance is used per hidden layer to avoid reusing the same
              module object across layers.

        Args:
            num_layers: Number of hidden layers.
            widths: Width(s) of hidden layers.
            activations: Activation function(s) to use.
            use_layer_norm: Whether to use layer normalization.
            dropout: Dropout probability.
            bias: Whether to include bias in linear layers.
        """
        # Ensure widths is a list
        if isinstance(widths, int):
            widths = [widths] * num_layers
        else:
            num_layers = len(widths)

        # Ensure activations is a list
        if not isinstance(activations, list):
            # Use deep copies so each layer gets its own module instance
            activations = [copy.deepcopy(activations) for _ in range(num_layers)]
        if len(activations) != num_layers:
            raise ValueError("Number of activations must match number of hidden layers")
        if any(not isinstance(act, nn.Module) for act in activations):
            raise ValueError("All activations must be instances of nn.Module")

        prev_dim = self.input_dim

        # Construct hidden layers
        for width, activation in zip(widths, activations):
            self.main_nn.append(nn.Linear(prev_dim, width, bias=bias))
            self.main_nn.append(activation)
            if use_layer_norm:
                self.main_nn.append(nn.LayerNorm(width))
            if dropout > 0:
                self.main_nn.append(nn.Dropout(dropout))
            prev_dim = width

        # Add output layer
        self.main_nn.append(nn.Linear(prev_dim, self.output_dim, bias=bias))


class GatedT5MSAEncoder(BaseModel):
    """
    Protein encoder that fuses ProtT5 and MSA Transformer features with a gate.

    Input layout:
        [ProtT5 vector | MSA vector | MSA mask]

    The MSA vector is projected into the ProtT5 hidden space, then a per-dimension
    gate chooses how much to trust MSA versus ProtT5:

        msa_proj = Wp(msa)
        gate = sigmoid(Wg([t5, msa_proj]))
        fused = gate * msa_proj + (1 - gate) * t5

    If the final mask value is 0, the encoder falls back to the ProtT5 vector.
    The fused vector is then passed through the regular Horizyn target MLP.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        t5_dim: int = 1024,
        msa_dim: int = 768,
        fusion_dim: int = 1024,
        num_layers: int = 1,
        widths: int | list[int] = 32,
        activations: nn.Module | list[nn.Module] = nn.ReLU(),
        use_layer_norm: bool = False,
        dropout: float = 0.0,
        bias: bool = True,
        normalise_output: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        expected_input_dim = t5_dim + msa_dim + 1
        if input_dim != expected_input_dim:
            raise ValueError(
                f"GatedT5MSAEncoder input_dim must equal t5_dim + msa_dim + 1. "
                f"Got input_dim={input_dim}, expected {expected_input_dim}."
            )
        if fusion_dim != t5_dim:
            raise ValueError(
                "This gate implementation expects fusion_dim to match t5_dim so "
                f"the fallback path is well-defined. Got fusion_dim={fusion_dim}, t5_dim={t5_dim}."
            )

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.t5_dim = t5_dim
        self.msa_dim = msa_dim
        self.fusion_dim = fusion_dim

        self.msa_projection = nn.Linear(msa_dim, fusion_dim, bias=bias)
        self.gate = nn.Linear(t5_dim + fusion_dim, fusion_dim, bias=bias)
        self.encoder = MLP(
            input_dim=fusion_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            widths=widths,
            activations=activations,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
            bias=bias,
            normalise_output=normalise_output,
        )

    @property
    def layers(self) -> nn.ModuleList:
        """Return layers for compatibility with normalization checks."""
        return nn.ModuleList([self.msa_projection, self.gate, *self.encoder.layers])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected rank-2 protein input, got shape={tuple(x.shape)}")
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected protein input dim {self.input_dim}, got {x.shape[1]}"
            )

        t5 = x[:, : self.t5_dim]
        msa = x[:, self.t5_dim : self.t5_dim + self.msa_dim]
        msa_mask = x[:, self.t5_dim + self.msa_dim : self.t5_dim + self.msa_dim + 1]
        msa_mask = msa_mask.clamp(0.0, 1.0)

        msa_projected = self.msa_projection(msa)
        gate = torch.sigmoid(self.gate(torch.cat([t5, msa_projected], dim=-1)))
        fused_with_msa = gate * msa_projected + (1.0 - gate) * t5
        fused = msa_mask * fused_with_msa + (1.0 - msa_mask) * t5

        return self.encoder(fused)


class ProjectedGatedT5MSAEncoder(BaseModel):
    """
    ProtT5 + MSA encoder with explicit projected-space gate.

    This follows the intended fusion pattern:

        h_msa = W_p msa
        g = sigmoid(W_g [h_t5, h_msa, |h_t5 - h_msa|, h_t5 * h_msa])
        h = g * h_msa + (1 - g) * h_t5

    The MSA projection is additionally regularized to align with the ProtT5
    space through an auxiliary loss exposed via ``auxiliary_loss``. The gate is
    initialized with a negative bias so training starts close to the ProtT5-only
    model while still allowing end-to-end retraining.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        t5_dim: int = 1024,
        msa_dim: int = 768,
        fusion_dim: int = 1024,
        projection_hidden_dim: int = 2048,
        gate_hidden_dim: int = 1024,
        gate_type: str = "vector",
        gate_bias_init: float = -3.0,
        alignment_loss_weight: float = 0.05,
        gate_l1_weight: float = 0.001,
        projection_dropout: float = 0.1,
        num_layers: int = 1,
        widths: int | list[int] = 32,
        activations: nn.Module | list[nn.Module] = nn.ReLU(),
        use_layer_norm: bool = False,
        dropout: float = 0.0,
        bias: bool = True,
        normalise_output: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        expected_input_dim = t5_dim + msa_dim + 1
        if input_dim != expected_input_dim:
            raise ValueError(
                f"ProjectedGatedT5MSAEncoder input_dim must equal t5_dim + msa_dim + 1. "
                f"Got input_dim={input_dim}, expected {expected_input_dim}."
            )
        if fusion_dim != t5_dim:
            raise ValueError(
                "ProjectedGatedT5MSAEncoder expects fusion_dim to match t5_dim. "
                f"Got fusion_dim={fusion_dim}, t5_dim={t5_dim}."
            )
        if gate_type not in {"vector", "scalar"}:
            raise ValueError(f"gate_type must be 'vector' or 'scalar', got {gate_type!r}")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.t5_dim = t5_dim
        self.msa_dim = msa_dim
        self.fusion_dim = fusion_dim
        self.gate_type = gate_type
        self.alignment_loss_weight = alignment_loss_weight
        self.gate_l1_weight = gate_l1_weight
        self._last_aux_loss: torch.Tensor | None = None
        self._last_aux_stats: dict[str, torch.Tensor] = {}

        self.t5_gate_norm = nn.LayerNorm(t5_dim)
        self.msa_norm = nn.LayerNorm(msa_dim)
        self.msa_projection = nn.Sequential(
            nn.Linear(msa_dim, projection_hidden_dim, bias=bias),
            nn.GELU(),
            nn.Dropout(projection_dropout),
            nn.LayerNorm(projection_hidden_dim),
            nn.Linear(projection_hidden_dim, fusion_dim, bias=bias),
        )
        self.projected_gate_norm = nn.LayerNorm(fusion_dim)

        gate_output_dim = fusion_dim if gate_type == "vector" else 1
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 4, gate_hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(projection_dropout),
            nn.Linear(gate_hidden_dim, gate_output_dim, bias=True),
        )
        final_gate_layer = self.gate[-1]
        if isinstance(final_gate_layer, nn.Linear):
            nn.init.zeros_(final_gate_layer.weight)
            nn.init.constant_(final_gate_layer.bias, gate_bias_init)

        self.encoder = MLP(
            input_dim=fusion_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            widths=widths,
            activations=activations,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
            bias=bias,
            normalise_output=normalise_output,
        )

    @property
    def layers(self) -> nn.ModuleList:
        return nn.ModuleList(
            [
                self.t5_gate_norm,
                self.msa_norm,
                *self.msa_projection,
                self.projected_gate_norm,
                *self.gate,
                *self.encoder.layers,
            ]
        )

    def auxiliary_loss(self) -> torch.Tensor | None:
        return self._last_aux_loss

    def auxiliary_stats(self) -> dict[str, torch.Tensor]:
        return self._last_aux_stats

    def _compute_auxiliary_loss(
        self,
        t5: torch.Tensor,
        msa_projected: torch.Tensor,
        gate: torch.Tensor,
        msa_mask: torch.Tensor,
    ) -> None:
        present = msa_mask.squeeze(-1) > 0.5
        if not torch.any(present):
            self._last_aux_loss = None
            self._last_aux_stats = {}
            return

        t5_aligned = self.t5_gate_norm(t5[present]).detach()
        msa_aligned = self.projected_gate_norm(msa_projected[present])
        alignment_loss = 1.0 - F.cosine_similarity(msa_aligned, t5_aligned, dim=-1).mean()
        gate_l1 = gate[present].mean()
        self._last_aux_loss = (
            self.alignment_loss_weight * alignment_loss + self.gate_l1_weight * gate_l1
        )
        self._last_aux_stats = {
            "msa_alignment_loss": alignment_loss.detach(),
            "msa_gate_mean": gate_l1.detach(),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected rank-2 protein input, got shape={tuple(x.shape)}")
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected protein input dim {self.input_dim}, got {x.shape[1]}"
            )

        t5 = x[:, : self.t5_dim]
        msa = x[:, self.t5_dim : self.t5_dim + self.msa_dim]
        msa_mask = x[:, self.t5_dim + self.msa_dim : self.t5_dim + self.msa_dim + 1]
        msa_mask = msa_mask.clamp(0.0, 1.0)

        msa_projected = self.msa_projection(self.msa_norm(msa))
        t5_for_gate = self.t5_gate_norm(t5)
        msa_for_gate = self.projected_gate_norm(msa_projected)
        gate_input = torch.cat(
            [
                t5_for_gate,
                msa_for_gate,
                torch.abs(t5_for_gate - msa_for_gate),
                t5_for_gate * msa_for_gate,
            ],
            dim=-1,
        )
        gate = torch.sigmoid(self.gate(gate_input))
        gate = gate * msa_mask
        fused = gate * msa_projected + (1.0 - gate) * t5
        self._compute_auxiliary_loss(t5, msa_projected, gate, msa_mask)

        return self.encoder(fused)


class DualProjectedGatedT5MSAEncoder(BaseModel):
    """
    ProtT5 + full-sequence MSA + pocket MSA encoder.

    Expected input layout:

        [ProtT5 | full MSA | full MSA mask | pocket MSA | pocket MSA mask]

    Full and pocket MSA views are projected separately into the ProtT5 space and
    receive separate gates. The final fusion is a stable weighted average:

        h = (t5 + g_full * h_full + g_pocket * h_pocket) / (1 + g_full + g_pocket)

    With negative gate bias this starts close to ProtT5-only behavior while
    allowing either MSA view to contribute when present.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        t5_dim: int = 1024,
        msa_dim: int = 768,
        fusion_dim: int = 1024,
        projection_hidden_dim: int = 2048,
        gate_hidden_dim: int = 1024,
        gate_type: str = "vector",
        gate_bias_init: float = -3.0,
        alignment_loss_weight: float = 0.05,
        gate_l1_weight: float = 0.001,
        projection_dropout: float = 0.1,
        num_layers: int = 1,
        widths: int | list[int] = 32,
        activations: nn.Module | list[nn.Module] = nn.ReLU(),
        use_layer_norm: bool = False,
        dropout: float = 0.0,
        bias: bool = True,
        normalise_output: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        expected_input_dim = t5_dim + msa_dim + 1 + msa_dim + 1
        if input_dim != expected_input_dim:
            raise ValueError(
                f"DualProjectedGatedT5MSAEncoder input_dim must equal "
                f"t5_dim + msa_dim + 1 + msa_dim + 1. "
                f"Got input_dim={input_dim}, expected {expected_input_dim}."
            )
        if fusion_dim != t5_dim:
            raise ValueError(
                "DualProjectedGatedT5MSAEncoder expects fusion_dim to match t5_dim. "
                f"Got fusion_dim={fusion_dim}, t5_dim={t5_dim}."
            )
        if gate_type not in {"vector", "scalar"}:
            raise ValueError(f"gate_type must be 'vector' or 'scalar', got {gate_type!r}")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.t5_dim = t5_dim
        self.msa_dim = msa_dim
        self.fusion_dim = fusion_dim
        self.gate_type = gate_type
        self.alignment_loss_weight = alignment_loss_weight
        self.gate_l1_weight = gate_l1_weight
        self._last_aux_loss: torch.Tensor | None = None
        self._last_aux_stats: dict[str, torch.Tensor] = {}

        self.t5_gate_norm = nn.LayerNorm(t5_dim)
        self.full_msa_norm = nn.LayerNorm(msa_dim)
        self.pocket_msa_norm = nn.LayerNorm(msa_dim)
        self.full_msa_projection = self._make_projection(
            msa_dim=msa_dim,
            projection_hidden_dim=projection_hidden_dim,
            fusion_dim=fusion_dim,
            bias=bias,
            dropout=projection_dropout,
        )
        self.pocket_msa_projection = self._make_projection(
            msa_dim=msa_dim,
            projection_hidden_dim=projection_hidden_dim,
            fusion_dim=fusion_dim,
            bias=bias,
            dropout=projection_dropout,
        )
        self.full_projected_gate_norm = nn.LayerNorm(fusion_dim)
        self.pocket_projected_gate_norm = nn.LayerNorm(fusion_dim)

        gate_output_dim = fusion_dim if gate_type == "vector" else 1
        self.full_gate = self._make_gate(
            fusion_dim=fusion_dim,
            gate_hidden_dim=gate_hidden_dim,
            gate_output_dim=gate_output_dim,
            dropout=projection_dropout,
            gate_bias_init=gate_bias_init,
        )
        self.pocket_gate = self._make_gate(
            fusion_dim=fusion_dim,
            gate_hidden_dim=gate_hidden_dim,
            gate_output_dim=gate_output_dim,
            dropout=projection_dropout,
            gate_bias_init=gate_bias_init,
        )

        self.encoder = MLP(
            input_dim=fusion_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            widths=widths,
            activations=activations,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
            bias=bias,
            normalise_output=normalise_output,
        )

    @staticmethod
    def _make_projection(
        msa_dim: int,
        projection_hidden_dim: int,
        fusion_dim: int,
        bias: bool,
        dropout: float,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(msa_dim, projection_hidden_dim, bias=bias),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(projection_hidden_dim),
            nn.Linear(projection_hidden_dim, fusion_dim, bias=bias),
        )

    @staticmethod
    def _make_gate(
        fusion_dim: int,
        gate_hidden_dim: int,
        gate_output_dim: int,
        dropout: float,
        gate_bias_init: float,
    ) -> nn.Sequential:
        gate = nn.Sequential(
            nn.Linear(fusion_dim * 4, gate_hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, gate_output_dim, bias=True),
        )
        final_gate_layer = gate[-1]
        if isinstance(final_gate_layer, nn.Linear):
            nn.init.zeros_(final_gate_layer.weight)
            nn.init.constant_(final_gate_layer.bias, gate_bias_init)
        return gate

    @property
    def layers(self) -> nn.ModuleList:
        return nn.ModuleList(
            [
                self.t5_gate_norm,
                self.full_msa_norm,
                self.pocket_msa_norm,
                *self.full_msa_projection,
                *self.pocket_msa_projection,
                self.full_projected_gate_norm,
                self.pocket_projected_gate_norm,
                *self.full_gate,
                *self.pocket_gate,
                *self.encoder.layers,
            ]
        )

    def auxiliary_loss(self) -> torch.Tensor | None:
        return self._last_aux_loss

    def auxiliary_stats(self) -> dict[str, torch.Tensor]:
        return self._last_aux_stats

    def _compute_gate(
        self,
        t5_for_gate: torch.Tensor,
        msa_projected: torch.Tensor,
        gate_norm: nn.LayerNorm,
        gate_module: nn.Sequential,
        msa_mask: torch.Tensor,
    ) -> torch.Tensor:
        msa_for_gate = gate_norm(msa_projected)
        gate_input = torch.cat(
            [
                t5_for_gate,
                msa_for_gate,
                torch.abs(t5_for_gate - msa_for_gate),
                t5_for_gate * msa_for_gate,
            ],
            dim=-1,
        )
        gate = torch.sigmoid(gate_module(gate_input))
        return gate * msa_mask

    def _view_auxiliary_terms(
        self,
        name: str,
        t5: torch.Tensor,
        msa_projected: torch.Tensor,
        gate: torch.Tensor,
        msa_mask: torch.Tensor,
        gate_norm: nn.LayerNorm,
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
        present = msa_mask.squeeze(-1) > 0.5
        if not torch.any(present):
            return None, {}

        t5_aligned = self.t5_gate_norm(t5[present]).detach()
        msa_aligned = gate_norm(msa_projected[present])
        alignment_loss = 1.0 - F.cosine_similarity(msa_aligned, t5_aligned, dim=-1).mean()
        gate_l1 = gate[present].mean()
        aux_loss = self.alignment_loss_weight * alignment_loss + self.gate_l1_weight * gate_l1
        stats = {
            f"{name}_msa_alignment_loss": alignment_loss.detach(),
            f"{name}_msa_gate_mean": gate_l1.detach(),
        }
        return aux_loss, stats

    def _compute_auxiliary_loss(
        self,
        t5: torch.Tensor,
        full_projected: torch.Tensor,
        pocket_projected: torch.Tensor,
        full_gate: torch.Tensor,
        pocket_gate: torch.Tensor,
        full_mask: torch.Tensor,
        pocket_mask: torch.Tensor,
    ) -> None:
        terms = []
        stats = {}

        full_loss, full_stats = self._view_auxiliary_terms(
            "full",
            t5,
            full_projected,
            full_gate,
            full_mask,
            self.full_projected_gate_norm,
        )
        if full_loss is not None:
            terms.append(full_loss)
            stats.update(full_stats)

        pocket_loss, pocket_stats = self._view_auxiliary_terms(
            "pocket",
            t5,
            pocket_projected,
            pocket_gate,
            pocket_mask,
            self.pocket_projected_gate_norm,
        )
        if pocket_loss is not None:
            terms.append(pocket_loss)
            stats.update(pocket_stats)

        if not terms:
            self._last_aux_loss = None
            self._last_aux_stats = {}
            return

        self._last_aux_loss = torch.stack(terms).mean()
        align_terms = [
            value
            for key, value in stats.items()
            if key.endswith("_msa_alignment_loss")
        ]
        gate_terms = [
            value
            for key, value in stats.items()
            if key.endswith("_msa_gate_mean")
        ]
        if align_terms:
            stats["msa_alignment_loss"] = torch.stack(align_terms).mean()
        if gate_terms:
            stats["msa_gate_mean"] = torch.stack(gate_terms).mean()
        self._last_aux_stats = stats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected rank-2 protein input, got shape={tuple(x.shape)}")
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected protein input dim {self.input_dim}, got {x.shape[1]}"
            )

        full_start = self.t5_dim
        full_mask_idx = full_start + self.msa_dim
        pocket_start = full_mask_idx + 1
        pocket_mask_idx = pocket_start + self.msa_dim

        t5 = x[:, : self.t5_dim]
        full_msa = x[:, full_start:full_mask_idx]
        full_mask = x[:, full_mask_idx : full_mask_idx + 1].clamp(0.0, 1.0)
        pocket_msa = x[:, pocket_start:pocket_mask_idx]
        pocket_mask = x[:, pocket_mask_idx : pocket_mask_idx + 1].clamp(0.0, 1.0)

        full_projected = self.full_msa_projection(self.full_msa_norm(full_msa))
        pocket_projected = self.pocket_msa_projection(self.pocket_msa_norm(pocket_msa))
        t5_for_gate = self.t5_gate_norm(t5)

        full_gate = self._compute_gate(
            t5_for_gate,
            full_projected,
            self.full_projected_gate_norm,
            self.full_gate,
            full_mask,
        )
        pocket_gate = self._compute_gate(
            t5_for_gate,
            pocket_projected,
            self.pocket_projected_gate_norm,
            self.pocket_gate,
            pocket_mask,
        )

        gate_total = full_gate + pocket_gate
        fused = (t5 + full_gate * full_projected + pocket_gate * pocket_projected) / (
            1.0 + gate_total
        )
        self._compute_auxiliary_loss(
            t5,
            full_projected,
            pocket_projected,
            full_gate,
            pocket_gate,
            full_mask,
            pocket_mask,
        )

        return self.encoder(fused)


class LateFusionT5MSAEncoder(BaseModel):
    """
    Safer ProtT5 + MSA fusion encoder for contrastive retrieval.

    Unlike GatedT5MSAEncoder, this keeps the ProtT5 target path intact and adds
    MSA only as a gated residual in the final embedding space:

        z_t5 = T5_MLP(t5)
        z_msa = MSA_MLP(msa)
        gate = sigmoid(Wg([z_t5, z_msa]))
        z = normalize(z_t5 + mask * gate * z_msa)

    The gate is initialized with a negative bias, so training starts very close
    to the ProtT5-only baseline and can learn to use MSA only where helpful.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        t5_dim: int = 1024,
        msa_dim: int = 768,
        num_layers: int = 1,
        widths: int | list[int] = 32,
        msa_num_layers: int = 1,
        msa_widths: int | list[int] = 1024,
        gate_bias_init: float = -4.0,
        fusion_mode: str = "residual",
        activations: nn.Module | list[nn.Module] = nn.ReLU(),
        use_layer_norm: bool = False,
        dropout: float = 0.0,
        bias: bool = True,
        normalise_output: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        expected_input_dim = t5_dim + msa_dim + 1
        if input_dim != expected_input_dim:
            raise ValueError(
                f"LateFusionT5MSAEncoder input_dim must equal t5_dim + msa_dim + 1. "
                f"Got input_dim={input_dim}, expected {expected_input_dim}."
            )
        if not normalise_output:
            raise ValueError("LateFusionT5MSAEncoder requires normalise_output=True")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.t5_dim = t5_dim
        self.msa_dim = msa_dim
        if fusion_mode not in {"residual", "convex"}:
            raise ValueError(
                f"fusion_mode must be 'residual' or 'convex', got {fusion_mode!r}"
            )
        self.fusion_mode = fusion_mode

        self.t5_encoder = MLP(
            input_dim=t5_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            widths=widths,
            activations=activations,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
            bias=bias,
            normalise_output=True,
        )
        self.msa_norm = nn.LayerNorm(msa_dim)
        self.msa_encoder = MLP(
            input_dim=msa_dim,
            output_dim=output_dim,
            num_layers=msa_num_layers,
            widths=msa_widths,
            activations=activations,
            use_layer_norm=True,
            dropout=dropout,
            bias=bias,
            normalise_output=True,
        )
        self.gate = nn.Linear(output_dim * 2, output_dim, bias=True)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_bias_init)
        self.output_normalize = NormalizeLayer(p=2, dim=-1)

    @property
    def layers(self) -> nn.ModuleList:
        """Return layers for compatibility with normalization checks."""
        return nn.ModuleList(
            [
                *self.t5_encoder.layers,
                self.msa_norm,
                *self.msa_encoder.layers,
                self.gate,
                self.output_normalize,
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected rank-2 protein input, got shape={tuple(x.shape)}")
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected protein input dim {self.input_dim}, got {x.shape[1]}"
            )

        t5 = x[:, : self.t5_dim]
        msa = x[:, self.t5_dim : self.t5_dim + self.msa_dim]
        msa_mask = x[:, self.t5_dim + self.msa_dim : self.t5_dim + self.msa_dim + 1]
        msa_mask = msa_mask.clamp(0.0, 1.0)

        t5_embed = self.t5_encoder(t5)
        msa_embed = self.msa_encoder(self.msa_norm(msa))
        gate = torch.sigmoid(self.gate(torch.cat([t5_embed, msa_embed], dim=-1)))
        if self.fusion_mode == "convex":
            effective_gate = msa_mask * gate
            fused = (1.0 - effective_gate) * t5_embed + effective_gate * msa_embed
        else:
            fused = t5_embed + msa_mask * gate * msa_embed

        return self.output_normalize(fused)


class DualViewReactionEncoder(BaseModel):
    """
    Reaction encoder with two retrieval views.

    The first view is architecture-compatible with the original Horizyn query
    encoder and can be initialized from a ProtT5-only checkpoint. The second view
    is trained to align reactions with the protein MSA view.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        base_output_dim: int = 512,
        aux_output_dim: int = 512,
        aux_weight: float = 0.35,
        num_layers: int = 1,
        widths: int | list[int] = 32,
        aux_num_layers: int = 1,
        aux_widths: int | list[int] = 2048,
        dynamic_aux_weight: bool = False,
        aux_weight_min: float = 0.05,
        aux_weight_max: float = 0.75,
        activations: nn.Module | list[nn.Module] = nn.ReLU(),
        use_layer_norm: bool = False,
        dropout: float = 0.0,
        bias: bool = True,
        normalise_output: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if output_dim != base_output_dim + aux_output_dim:
            raise ValueError(
                f"output_dim must equal base_output_dim + aux_output_dim. "
                f"Got output_dim={output_dim}, base={base_output_dim}, aux={aux_output_dim}."
            )
        if not (0.0 < aux_weight < 1.0):
            raise ValueError(f"aux_weight must be in (0, 1), got {aux_weight}")
        if not (0.0 <= aux_weight_min < aux_weight_max <= 1.0):
            raise ValueError(
                "aux_weight_min and aux_weight_max must satisfy "
                f"0 <= min < max <= 1. Got min={aux_weight_min}, max={aux_weight_max}."
            )
        if not (aux_weight_min <= aux_weight <= aux_weight_max):
            raise ValueError(
                "aux_weight must be inside [aux_weight_min, aux_weight_max]. "
                f"Got aux_weight={aux_weight}, min={aux_weight_min}, max={aux_weight_max}."
            )
        if not normalise_output:
            raise ValueError("DualViewReactionEncoder requires normalise_output=True")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.base_output_dim = base_output_dim
        self.aux_output_dim = aux_output_dim
        self.aux_weight = aux_weight
        self.dynamic_aux_weight = dynamic_aux_weight
        self.aux_weight_min = aux_weight_min
        self.aux_weight_max = aux_weight_max

        self.base_encoder = MLP(
            input_dim=input_dim,
            output_dim=base_output_dim,
            num_layers=num_layers,
            widths=widths,
            activations=activations,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
            bias=bias,
            normalise_output=True,
        )
        self.aux_encoder = MLP(
            input_dim=input_dim,
            output_dim=aux_output_dim,
            num_layers=aux_num_layers,
            widths=aux_widths,
            activations=activations,
            use_layer_norm=True,
            dropout=dropout,
            bias=bias,
            normalise_output=True,
        )
        if dynamic_aux_weight:
            self.aux_gate = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, 1))
            prior = (aux_weight - aux_weight_min) / (aux_weight_max - aux_weight_min)
            prior = min(max(prior, 1e-6), 1.0 - 1e-6)
            with torch.no_grad():
                self.aux_gate[-1].weight.zero_()
                self.aux_gate[-1].bias.fill_(torch.logit(torch.tensor(prior)).item())
        else:
            self.aux_gate = None
        self.output_normalize = NormalizeLayer(p=2, dim=-1)

    @property
    def layers(self) -> nn.ModuleList:
        gate_layers = [] if self.aux_gate is None else list(self.aux_gate)
        return nn.ModuleList(
            [
                *self.base_encoder.layers,
                *self.aux_encoder.layers,
                *gate_layers,
                self.output_normalize,
            ]
        )

    def _aux_weights(self, x: torch.Tensor) -> torch.Tensor:
        if self.aux_gate is None:
            return torch.full(
                (x.shape[0], 1),
                self.aux_weight,
                dtype=x.dtype,
                device=x.device,
            )
        gate = torch.sigmoid(self.aux_gate(x))
        return self.aux_weight_min + (self.aux_weight_max - self.aux_weight_min) * gate

    def encode_views(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        base = self.base_encoder(x)
        aux = self.aux_encoder(x)
        aux_weight = self._aux_weights(x)
        return {"base": base, "aux": aux, "aux_weight": aux_weight}

    def combine_views(self, views: dict[str, torch.Tensor]) -> torch.Tensor:
        aux_weight = views["aux_weight"]
        base_scale = torch.sqrt(1.0 - aux_weight)
        aux_scale = torch.sqrt(aux_weight)
        return self.output_normalize(
            torch.cat([base_scale * views["base"], aux_scale * views["aux"]], dim=-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.combine_views(self.encode_views(x))


class DualViewT5MSAEncoder(BaseModel):
    """
    Protein encoder with separate ProtT5 and MSA retrieval views.

    The first view is architecture-compatible with the original Horizyn target
    encoder and can be initialized from a ProtT5-only checkpoint. The second view
    encodes MSA features. The final dot product is a weighted sum of the ProtT5
    and MSA-view similarities.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        t5_dim: int = 1024,
        msa_dim: int = 768,
        base_output_dim: int = 512,
        aux_output_dim: int = 512,
        aux_weight: float = 0.35,
        num_layers: int = 1,
        widths: int | list[int] = 32,
        msa_num_layers: int = 1,
        msa_widths: int | list[int] = 2048,
        activations: nn.Module | list[nn.Module] = nn.ReLU(),
        use_layer_norm: bool = False,
        dropout: float = 0.0,
        bias: bool = True,
        normalise_output: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        expected_input_dim = t5_dim + msa_dim + 1
        if input_dim != expected_input_dim:
            raise ValueError(
                f"DualViewT5MSAEncoder input_dim must equal t5_dim + msa_dim + 1. "
                f"Got input_dim={input_dim}, expected {expected_input_dim}."
            )
        if output_dim != base_output_dim + aux_output_dim:
            raise ValueError(
                f"output_dim must equal base_output_dim + aux_output_dim. "
                f"Got output_dim={output_dim}, base={base_output_dim}, aux={aux_output_dim}."
            )
        if not (0.0 < aux_weight < 1.0):
            raise ValueError(f"aux_weight must be in (0, 1), got {aux_weight}")
        if not normalise_output:
            raise ValueError("DualViewT5MSAEncoder requires normalise_output=True")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.t5_dim = t5_dim
        self.msa_dim = msa_dim
        self.base_output_dim = base_output_dim
        self.aux_output_dim = aux_output_dim
        self.aux_weight = aux_weight

        self.t5_encoder = MLP(
            input_dim=t5_dim,
            output_dim=base_output_dim,
            num_layers=num_layers,
            widths=widths,
            activations=activations,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
            bias=bias,
            normalise_output=True,
        )
        self.msa_norm = nn.LayerNorm(msa_dim)
        self.msa_encoder = MLP(
            input_dim=msa_dim,
            output_dim=aux_output_dim,
            num_layers=msa_num_layers,
            widths=msa_widths,
            activations=activations,
            use_layer_norm=True,
            dropout=dropout,
            bias=bias,
            normalise_output=True,
        )
        self.output_normalize = NormalizeLayer(p=2, dim=-1)

    @property
    def layers(self) -> nn.ModuleList:
        return nn.ModuleList(
            [
                *self.t5_encoder.layers,
                self.msa_norm,
                *self.msa_encoder.layers,
                self.output_normalize,
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.combine_views(self.encode_views(x))

    def encode_views(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 2:
            raise ValueError(f"Expected rank-2 protein input, got shape={tuple(x.shape)}")
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected protein input dim {self.input_dim}, got {x.shape[1]}"
            )

        t5 = x[:, : self.t5_dim]
        msa = x[:, self.t5_dim : self.t5_dim + self.msa_dim]
        msa_mask = x[:, self.t5_dim + self.msa_dim : self.t5_dim + self.msa_dim + 1]
        msa_mask = msa_mask.clamp(0.0, 1.0)

        t5_embed = self.t5_encoder(t5)
        msa_embed = self.msa_encoder(self.msa_norm(msa))
        return {"base": t5_embed, "aux": msa_embed, "msa_mask": msa_mask}

    def combine_views(self, views: dict[str, torch.Tensor]) -> torch.Tensor:
        base_scale = (1.0 - self.aux_weight) ** 0.5
        aux_scale = self.aux_weight**0.5

        # If MSA is missing, the auxiliary view is zeroed. This is rare for data1
        # and avoids injecting arbitrary MSA-view signal.
        aux = aux_scale * views["msa_mask"] * views["aux"]
        return self.output_normalize(torch.cat([base_scale * views["base"], aux], dim=-1))


class DualContrastiveModel(BaseModel):
    """
    Dual encoder contrastive learning model for Horizyn.

    This model uses separate encoders for query (reaction) and target (protein)
    inputs, producing normalized embeddings for contrastive learning. This is the
    core architecture of the Horizyn SOTA model.

    The model enforces that both encoders output normalized embeddings by checking
    for a NormalizeLayer as the final layer in each encoder.

    Notes:
        - Dict inputs: Dictionary inputs are forwarded to encoders via keyword
          arguments (i.e., encoder(**inputs)). This only works if the encoder
          classes accept those keyword arguments. The default `MLP` expects a
          tensor input and does not consume dicts.
        - Normalization enforcement: When `enforce_normalisation=True`, both
          encoders must end with a `NormalizeLayer`. Custom encoders should append
          `NormalizeLayer` as the last layer or disable enforcement explicitly.
    """

    def __init__(
        self,
        query_encoder_kwargs: dict[str, Any],
        target_encoder_kwargs: dict[str, Any],
        query_encoder: type[BaseModel] = MLP,
        target_encoder: type[BaseModel] = MLP,
        enforce_normalisation: bool = True,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Initialize the DualContrastiveModel.

        Args:
            query_encoder_kwargs: Keyword arguments for query encoder (reactions).
            target_encoder_kwargs: Keyword arguments for target encoder (proteins).
            query_encoder: Query encoder class (default: MLP).
            target_encoder: Target encoder class (default: MLP).
            enforce_normalisation: Whether to enforce that encoders have normalized outputs.
            *args: Additional arguments (must be empty).
            **kwargs: Additional keyword arguments (must be empty).

        Raises:
            ValueError: If enforce_normalisation is True and encoders don't have
                NormalizeLayer as final layer.

        Example:
            >>> # SOTA configuration
            >>> model = DualContrastiveModel(
            ...     query_encoder_kwargs={
            ...         "input_dim": 2048,
            ...         "output_dim": 512,
            ...         "num_layers": 1,
            ...         "widths": 4096,
            ...         "normalise_output": True,
            ...     },
            ...     target_encoder_kwargs={
            ...         "input_dim": 1024,
            ...         "output_dim": 512,
            ...         "num_layers": 1,
            ...         "widths": 4096,
            ...         "normalise_output": True,
            ...     },
            ... )
        """
        super().__init__(*args, **kwargs)
        self.query_encoder = query_encoder(**query_encoder_kwargs)
        self.target_encoder = target_encoder(**target_encoder_kwargs)

        # Validate that encoders have normalized outputs
        if enforce_normalisation:
            if not isinstance(self.query_encoder, BaseModel):
                raise ValueError("query_encoder must be a BaseModel instance")
            if not isinstance(self.target_encoder, BaseModel):
                raise ValueError("target_encoder must be a BaseModel instance")

            # Check that query encoder has NormalizeLayer as last layer
            if len(self.query_encoder.layers) == 0 or not isinstance(
                self.query_encoder.layers[-1], NormalizeLayer
            ):
                raise ValueError(
                    "query_encoder must have a NormalizeLayer as its last layer. "
                    "Set normalise_output=True in query_encoder_kwargs."
                )

            # Check that target encoder has NormalizeLayer as last layer
            if len(self.target_encoder.layers) == 0 or not isinstance(
                self.target_encoder.layers[-1], NormalizeLayer
            ):
                raise ValueError(
                    "target_encoder must have a NormalizeLayer as its last layer. "
                    "Set normalise_output=True in target_encoder_kwargs."
                )

    def forward(
        self, query_inputs: dict | torch.Tensor, target_inputs: dict | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the dual contrastive model.

        Encodes both query and target inputs through their respective encoders.
        Supports both tensor and dict inputs for flexibility.

        Args:
            query_inputs: Input for query encoder (reactions).
                Can be a tensor or dict of tensors.
            target_inputs: Input for target encoder (proteins).
                Can be a tensor or dict of tensors.

        Returns:
            Tuple of (query_embeddings, target_embeddings), both normalized.

        Raises:
            ValueError: If outputs are not rank-2 or if feature dimensions differ.

        Example:
            >>> query_fps = torch.randn(16, 2048)  # Reaction fingerprints
            >>> target_embs = torch.randn(16, 1024)  # Protein T5 embeddings
            >>> query_out, target_out = model(query_fps, target_embs)
            >>> query_out.shape, target_out.shape
            (torch.Size([16, 512]), torch.Size([16, 512]))
        """
        # Encode query inputs
        query = (
            self.query_encoder(**query_inputs)
            if isinstance(query_inputs, dict)
            else self.query_encoder(query_inputs)
        )

        # Encode target inputs
        target = (
            self.target_encoder(**target_inputs)
            if isinstance(target_inputs, dict)
            else self.target_encoder(target_inputs)
        )

        # Validate output ranks
        if query.ndim != 2 or target.ndim != 2:
            raise ValueError(
                f"Encoders must return rank-2 tensors (batch, features): "
                f"query.shape={tuple(query.shape)}, target.shape={tuple(target.shape)}"
            )

        # Validate output dimensions match
        if query.shape[1] != target.shape[1]:
            raise ValueError(
                f"Query and target encoder output shape mismatch: "
                f"query.shape={tuple(query.shape)} != target.shape={tuple(target.shape)}"
            )

        return query, target
