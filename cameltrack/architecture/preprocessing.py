import logging
from typing import Sequence, Optional

import torch
import torch.nn as nn

from cameltrack.architecture.base_module import Module

log = logging.getLogger(__name__)


class Preprocessing(Module):
    """Base interface for preprocessing blocks.

    Contract:
        tracks, dets = preprocessing(tracks, dets)

    Input:
        tracks.tokens: dict[str, Tensor], each [B, N, D]
        dets.tokens:   dict[str, Tensor], each [B, M, D]

    Output:
        tracks.cue_ctx: [B, N, K, D]
        dets.cue_ctx:   [B, M, K, D]
        tracks.cue_names / dets.cue_names should be set.
    """

    cue_dim: int = 1024

    def forward(self, tracks, dets):
        raise NotImplementedError


# ======================================================================================================================
# ======================================================================================================================
# METHOD 1: RAW
# ======================================================================================================================
# ======================================================================================================================
class RAW(Preprocessing):
    """RAW cue preprocessing.

    This is the direct stacking baseline.

    Input:
        tracks.tokens[name]: [B, N, D]
        dets.tokens[name]:   [B, M, D]

    Output:
        tracks.cue_ctx: [B, N, K, D]
        dets.cue_ctx:   [B, M, K, D]
    """

    def __init__(
        self,
        cue_names: Sequence[str] = ("app_encoder", "kp_encoder", "bbox_encoder"),
        *args,
        **kwargs,
    ):
        super().__init__()
        self.cue_names = tuple(cue_names)
        self.num_cues = len(self.cue_names)

    def _stack_cues(self, token_dict):
        """Stack cue-token dict into [B, E, K, D] following self.cue_names order."""
        missing = [name for name in self.cue_names if name not in token_dict]
        if missing:
            raise KeyError(
                f"Missing cue tokens: {missing}. "
                f"Found: {list(token_dict.keys())}"
            )

        return torch.stack(
            [token_dict[name] for name in self.cue_names],
            dim=2,
        )  # [B, E, K, D]

    def forward(self, tracks, dets):
        tracks.cue_ctx = self._stack_cues(tracks.tokens)
        dets.cue_ctx = self._stack_cues(dets.tokens)

        tracks.cue_names = self.cue_names
        dets.cue_names = self.cue_names

        return tracks, dets


# ======================================================================================================================
# ======================================================================================================================
# MODEL 2: CWT
# ======================================================================================================================
# ======================================================================================================================
class CWT(Preprocessing):
    """Cue-Wise Transformer preprocessing.

    CWT uses one independent shallow self-attention transformer per cue.

    For each cue k, it builds the sequence:

        [CLS_k, track_1^k, ..., track_N^k, det_1^k, ..., det_M^k]

    and applies a cue-specific TransformerEncoder.

    The cue-specific [CLS] token is intended to summarize the batch-level
    distribution of that cue over the current track/detection candidate set.

    Input:
        tracks.tokens[name]: [B, N, D]
        dets.tokens[name]:   [B, M, D]

    Output:
        tracks.cue_ctx: [B, N, K, D]
        dets.cue_ctx:   [B, M, K, D]

    Extra diagnostic output:
        tracks.cue_cls: [B, K, D]
        dets.cue_cls:   [B, K, D]

    Note:
        tracks.cue_cls and dets.cue_cls are the same tensor, because the CLS
        summarizes the joint track+detection set for each cue.
    """

    def __init__(
        self,
        cue_dim: int = 1024,
        cue_names: Sequence[str] = ("app_encoder", "kp_encoder", "bbox_encoder"),
        n_heads: int = 8,
        n_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        use_side_embeddings: bool = True,
        checkpoint_path: Optional[str] = None,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.cue_dim = cue_dim
        self.cue_names = tuple(cue_names)
        self.num_cues = len(self.cue_names)
        self.use_side_embeddings = use_side_embeddings

        # One learnable CLS token per cue.
        self.cls_tokens = nn.Parameter(
            torch.zeros(1, self.num_cues, 1, cue_dim)
        )

        # Optional side embeddings allow the transformer to distinguish
        # track tokens from detection tokens.
        if self.use_side_embeddings:
            self.track_side = nn.Parameter(
                torch.zeros(1, self.num_cues, 1, cue_dim)
            )
            self.det_side = nn.Parameter(
                torch.zeros(1, self.num_cues, 1, cue_dim)
            )
        else:
            self.track_side = None
            self.det_side = None

        # Three parallel transformers if cue_names has length 3.
        # More generally: one independent transformer per cue.
        self.encoders = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=cue_dim,
                    nhead=n_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=False,
                ),
                num_layers=n_layers,
            )
            for _ in range(self.num_cues)
        ])

        self.out_norms = nn.ModuleList([
            nn.LayerNorm(cue_dim)
            for _ in range(self.num_cues)
        ])

        self.last_debug = {}

        self._reset_parameters()
        self.init_weights(checkpoint_path=checkpoint_path, module_name="CWT")

    def _reset_parameters(self):
        nn.init.trunc_normal_(self.cls_tokens, std=0.02)

        if self.use_side_embeddings:
            nn.init.trunc_normal_(self.track_side, std=0.02)
            nn.init.trunc_normal_(self.det_side, std=0.02)

    def _check_tokens(self, tracks, dets):
        missing_tracks = [name for name in self.cue_names if name not in tracks.tokens]
        missing_dets = [name for name in self.cue_names if name not in dets.tokens]

        if missing_tracks:
            raise KeyError(
                f"Missing track cue tokens: {missing_tracks}. "
                f"Found: {list(tracks.tokens.keys())}"
            )

        if missing_dets:
            raise KeyError(
                f"Missing detection cue tokens: {missing_dets}. "
                f"Found: {list(dets.tokens.keys())}"
            )

    def forward(self, tracks, dets):
        self._check_tokens(tracks, dets)

        track_mask = tracks.masks  # [B, N]
        det_mask = dets.masks      # [B, M]

        B, N = track_mask.shape
        M = det_mask.shape[1]
        device = track_mask.device

        track_outputs = []
        det_outputs = []
        cls_outputs = []

        for k, cue_name in enumerate(self.cue_names):
            track_tokens = tracks.tokens[cue_name]  # [B, N, D]
            det_tokens = dets.tokens[cue_name]      # [B, M, D]

            if track_tokens.shape[-1] != self.cue_dim:
                raise ValueError(
                    f"Track cue '{cue_name}' has dim {track_tokens.shape[-1]}, "
                    f"expected {self.cue_dim}."
                )

            if det_tokens.shape[-1] != self.cue_dim:
                raise ValueError(
                    f"Detection cue '{cue_name}' has dim {det_tokens.shape[-1]}, "
                    f"expected {self.cue_dim}."
                )

            if self.use_side_embeddings:
                track_tokens = track_tokens + self.track_side[:, k, :, :]
                det_tokens = det_tokens + self.det_side[:, k, :, :]

            entity_tokens = torch.cat(
                [track_tokens, det_tokens],
                dim=1,
            )  # [B, N + M, D]

            entity_mask = torch.cat(
                [track_mask, det_mask],
                dim=1,
            )  # [B, N + M]

            cls = self.cls_tokens[:, k, :, :].expand(
                B,
                1,
                self.cue_dim,
            )  # [B, 1, D]

            x = torch.cat(
                [cls, entity_tokens],
                dim=1,
            )  # [B, 1 + N + M, D]

            cls_mask = torch.ones(
                B,
                1,
                dtype=torch.bool,
                device=device,
            )  # [B, 1]

            full_mask = torch.cat(
                [cls_mask, entity_mask],
                dim=1,
            )  # [B, 1 + N + M]

            x = self.encoders[k](
                x,
                src_key_padding_mask=~full_mask,
            )  # [B, 1 + N + M, D]

            x = self.out_norms[k](x)

            cls_out = x[:, 0, :]       # [B, D]
            entity_out = x[:, 1:, :]   # [B, N + M, D]

            # Remove invalid padded entity outputs.
            entity_out = entity_out * entity_mask.unsqueeze(-1)

            track_out = entity_out[:, :N, :]       # [B, N, D]
            det_out = entity_out[:, N:N + M, :]    # [B, M, D]

            track_outputs.append(track_out)
            det_outputs.append(det_out)
            cls_outputs.append(cls_out)

        tracks.cue_ctx = torch.stack(
            track_outputs,
            dim=2,
        )  # [B, N, K, D]

        dets.cue_ctx = torch.stack(
            det_outputs,
            dim=2,
        )  # [B, M, K, D]

        cue_cls = torch.stack(
            cls_outputs,
            dim=1,
        )  # [B, K, D]

        tracks.cue_cls = cue_cls
        dets.cue_cls = cue_cls

        tracks.cue_names = self.cue_names
        dets.cue_names = self.cue_names

        self.last_debug = {
            "track_cue_ctx": tracks.cue_ctx.detach(),
            "det_cue_ctx": dets.cue_ctx.detach(),
            "cue_cls": cue_cls.detach(),
            "cue_names": self.cue_names,
        }

        return tracks, dets