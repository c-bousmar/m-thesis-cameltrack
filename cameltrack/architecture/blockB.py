import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cameltrack.architecture.base_module import Module
from cameltrack.utils.similarity_metrics import similarity_metrics

log = logging.getLogger(__name__)


class BlockB(Module):
    """Base interface for association-scoring blocks.

    Contract:
        td_sim_matrix = blockB(tracks, dets)
        returns [B, N, M] scores where larger means more likely association.

    Optional training contract:
        If BlockB is learned and outputs logits, it should set:
            tracks.pair_logits / dets.pair_logits: [B, N, M]
            tracks.pair_scores / dets.pair_scores: [B, N, M]
        so camelv2.py can reuse the existing pairwise BCE loss.
    """
    cue_dim: int = 1024

    def _requires_attrs(self, tracks, dets):
        required = [
            (tracks, "embs"),
            (dets, "embs"),
            (tracks, "cue_ctx"),
            (dets, "cue_ctx"),
        ]
        missing = [name for obj, name in required if not hasattr(obj, name)]
        if missing:
            raise RuntimeError(f"BlockB required inputs. Missing attributes: {missing}")

    def forward(self, tracks, dets):
        raise NotImplementedError


# ======================================================================================================================
# ======================================================================================================================
# ES
# ======================================================================================================================
# ======================================================================================================================
class EuclideanSimilarity(BlockB):
    """Default uses CAMELTrack's `norm_euclidean` similarity metric, matching the
    original CAMEL default configuration. If tracks.embs/dets.embs are dicts,
    one similarity matrix is computed per entry and averaged.
    """

    def __init__(
        self,
        sim_strat: str = "norm_euclidean",
        cue_dim: int = 1024,
        checkpoint_path: Optional[str] = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        if sim_strat not in similarity_metrics:
            raise NotImplementedError(f"Unknown similarity strategy: {sim_strat}")
        self.sim_strat = sim_strat
        self.similarity_metric = similarity_metrics[sim_strat]
        self.cue_dim = cue_dim
        self.init_weights(checkpoint_path=checkpoint_path, module_name="embedding_euclidean_blockB")

    def forward(self, tracks, dets):
        self._requires_attrs(tracks, dets)
        if isinstance(tracks.embs, dict):
            sims = []
            for key in tracks.embs.keys():
                if key not in dets.embs:
                    raise KeyError(f"dets.embs does not contain key '{key}'")
                sims.append(self.similarity_metric(tracks.embs[key], tracks.masks, dets.embs[key], dets.masks))
            td_sim_matrix = torch.stack(sims, dim=0).mean(dim=0)
        else:
            td_sim_matrix = self.similarity_metric(tracks.embs, tracks.masks, dets.embs, dets.masks)
        return td_sim_matrix

# ======================================================================================================================
# ======================================================================================================================
# CWP-MLP
# ======================================================================================================================
# ======================================================================================================================
class CWP_MLP(BlockB):
    """CWP-MLP pairwise association scorer.
    Expected BlockA input:
        CWP-MLP must have already attached:
            tracks.cls_ctx:  [B, N, D]
            dets.cls_ctx:    [B, M, D]
            tracks.cue_ctx: [B, N, K, D]
            dets.cue_ctx:   [B, M, K, D]

    This implements the CWP-MLP scoring stages:
        global CLS outputs -> pair_ctx
        contextualized cue outputs -> cue keys
        dot(pair_ctx, cue keys) -> cue weights
        weighted cue summaries -> pair feature
        MLP -> pair logits/scores
    """

    def __init__(
        self,
        cue_dim: int = 1024,
        pair_hidden_dim: int = 1024,
        dropout: float = 0.1,
        checkpoint_path: Optional[str] = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.cue_dim = cue_dim

        self.track_pair_proj = nn.Linear(cue_dim, cue_dim)
        self.det_pair_proj = nn.Linear(cue_dim, cue_dim)
        self.track_cue_key = nn.Linear(cue_dim, cue_dim)
        self.det_cue_key = nn.Linear(cue_dim, cue_dim)
        self.pair_norm = nn.LayerNorm(cue_dim)

        self.score_head = nn.Sequential(
            nn.Linear(4 * cue_dim, pair_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pair_hidden_dim, 1),
        )

        self.last_debug = {}
        self.init_weights(checkpoint_path=checkpoint_path, module_name="cwp_mlp_blockB")

    def forward(self, tracks, dets):
        self._requires_attrs(tracks, dets)

        track_cls_ctx = tracks.embs         # [B, N, D]
        det_cls_ctx = dets.embs             # [B, M, D]
        track_cue_ctx = tracks.cue_ctx    # [B, N, K, D]
        det_cue_ctx = dets.cue_ctx        # [B, M, K, D]

        track_mask = tracks.masks
        det_mask = dets.masks
        valid_pairs = track_mask.unsqueeze(2) & det_mask.unsqueeze(1)

        B, N, D = track_cls_ctx.shape
        M = det_cls_ctx.shape[1]
        if N == 0 or M == 0:
            pair_logits = torch.zeros(B, N, M, device=track_cls_ctx.device)
            pair_scores = torch.full((B, N, M), -float("inf"), device=track_cls_ctx.device)
            tracks.pair_logits = pair_logits
            tracks.pair_scores = pair_scores
            dets.pair_logits = pair_logits
            dets.pair_scores = pair_scores
            self.last_debug = {}
            return pair_scores

        pair_ctx = self.pair_norm(
            self.track_pair_proj(track_cls_ctx).unsqueeze(2)
            + self.det_pair_proj(det_cls_ctx).unsqueeze(1)
        )  # [B, N, M, D]

        track_keys = self.track_cue_key(track_cue_ctx)  # [B, N, K, D]
        det_keys = self.det_cue_key(det_cue_ctx)        # [B, M, K, D]

        track_cue_logits = torch.einsum("bnmd,bnkd->bnmk", pair_ctx, track_keys) / math.sqrt(self.cue_dim)
        det_cue_logits = torch.einsum("bnmd,bmkd->bnmk", pair_ctx, det_keys) / math.sqrt(self.cue_dim)

        track_cue_weights = F.softmax(track_cue_logits, dim=-1)
        det_cue_weights = F.softmax(det_cue_logits, dim=-1)

        track_weighted_summary = torch.einsum("bnmk,bnkd->bnmd", track_cue_weights, track_cue_ctx)
        det_weighted_summary = torch.einsum("bnmk,bmkd->bnmd", det_cue_weights, det_cue_ctx)

        pair_feat = torch.cat(
            [
                pair_ctx,
                track_weighted_summary,
                det_weighted_summary,
                torch.abs(track_weighted_summary - det_weighted_summary),
            ],
            dim=-1,
        )  # [B, N, M, 4D]

        raw_pair_logits = self.score_head(pair_feat).squeeze(-1)
        raw_pair_scores = torch.sigmoid(raw_pair_logits)
        
        pair_logits = raw_pair_logits.masked_fill(~valid_pairs, 0.0)
        pair_scores = raw_pair_scores.masked_fill(~valid_pairs, -float("inf"))

        track_cue_weights = track_cue_weights * valid_pairs.unsqueeze(-1)
        det_cue_weights = det_cue_weights * valid_pairs.unsqueeze(-1)

        tracks.pair_logits = pair_logits
        tracks.pair_scores = pair_scores
        dets.pair_logits = pair_logits
        dets.pair_scores = pair_scores

        self.last_debug = {
            "pair_ctx": pair_ctx.detach(),
            "track_cue_weights": track_cue_weights.detach(),
            "det_cue_weights": det_cue_weights.detach(),
            "track_weighted_summary": track_weighted_summary.detach(),
            "det_weighted_summary": det_weighted_summary.detach(),
            "pair_scores": pair_scores.detach(),
            "cue_names": getattr(tracks, "cue_names", None),
        }
        return pair_scores


# ======================================================================================================================
# ======================================================================================================================
# EP-MLP
# ======================================================================================================================
# ======================================================================================================================
class P_MLP(BlockB):
    def __init__(
        self,
        cue_dim: int = 1024,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
        use_absdiff: bool = True,
        use_product: bool = True,
        embedding_key: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.cue_dim = cue_dim
        self.use_absdiff = use_absdiff
        self.use_product = use_product
        self.embedding_key = embedding_key

        in_dim = 2 * cue_dim
        if use_absdiff:
            in_dim += cue_dim
        if use_product:
            in_dim += cue_dim

        self.score_head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.last_debug = {}
        self.init_weights(checkpoint_path=checkpoint_path, module_name="p_mlp_blockB")

    def _get_embs(self, tracks, dets):
        if isinstance(tracks.embs, dict):
            key = self.embedding_key if self.embedding_key is not None else next(iter(tracks.embs.keys()))
            if key not in tracks.embs or key not in dets.embs:
                raise KeyError(f"Embedding key '{key}' not found in tracks.embs/dets.embs")
            return tracks.embs[key], dets.embs[key]
        return tracks.embs, dets.embs

    def _make_pair_features(self, track_embs, det_embs):
        t = track_embs.unsqueeze(2)  # [B, N, 1, D]
        d = det_embs.unsqueeze(1)    # [B, 1, M, D]
        t = t.expand(-1, -1, det_embs.shape[1], -1)
        d = d.expand(-1, track_embs.shape[1], -1, -1)

        feats = [t, d]
        if self.use_absdiff:
            feats.append(torch.abs(t - d))
        if self.use_product:
            feats.append(t * d)
        return torch.cat(feats, dim=-1)

    def forward(self, tracks, dets):
        self._requires_attrs(tracks, dets)
        track_embs, det_embs = self._get_embs(tracks, dets)
        pair_feat = self._make_pair_features(track_embs, det_embs)

        raw_pair_logits = self.score_head(pair_feat).squeeze(-1)
        raw_pair_scores = torch.sigmoid(raw_pair_logits)

        valid_pairs = tracks.masks.unsqueeze(2) & dets.masks.unsqueeze(1)
        pair_scores = raw_pair_scores.masked_fill(~valid_pairs, -float("inf"))
        pair_logits = raw_pair_logits.masked_fill(~valid_pairs, 0.0)

        tracks.pair_logits = pair_logits
        tracks.pair_scores = pair_scores
        dets.pair_logits = pair_logits
        dets.pair_scores = pair_scores

        self.last_debug = {"pair_scores": pair_scores.detach()}
        return pair_scores


# ======================================================================================================================
# ======================================================================================================================
# EP-T
# ======================================================================================================================
# ======================================================================================================================
class _TransformerLayer(nn.Module):
    def __init__(self, dim: int, n_heads: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x

class P_Transformer(BlockB):
    """Paired-Transformer that outputs one scalar logit per track/detection pair.

    For each pair (track_i, det_j), it builds a tiny token sequence:
        [PAIR_CLS, track_emb, det_emb, optional absdiff, optional product]
    """
    def __init__(
        self,
        cue_dim: int = 1024,
        n_heads: int = 8,
        n_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        use_absdiff_token: bool = True,
        use_product_token: bool = True,
        embedding_key: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.cue_dim = cue_dim
        self.use_absdiff_token = use_absdiff_token
        self.use_product_token = use_product_token
        self.embedding_key = embedding_key

        self.n_tokens = 3 + int(use_absdiff_token) + int(use_product_token)
        self.pair_cls = nn.Parameter(torch.zeros(1, 1, cue_dim))
        self.token_type = nn.Parameter(torch.zeros(1, self.n_tokens, cue_dim))

        self.layers = nn.ModuleList([
            _TransformerLayer(cue_dim, n_heads, dim_feedforward, dropout)
            for _ in range(n_layers)
        ])
        self.score_head = nn.Sequential(
            nn.LayerNorm(cue_dim),
            nn.Linear(cue_dim, 1),
        )

        self.last_debug = {}
        self.init_weights(checkpoint_path=checkpoint_path, module_name="p_transformer_blockB")

    def _get_embs(self, tracks, dets):
        if isinstance(tracks.embs, dict):
            key = self.embedding_key if self.embedding_key is not None else next(iter(tracks.embs.keys()))
            if key not in tracks.embs or key not in dets.embs:
                raise KeyError(f"Embedding key '{key}' not found in tracks.embs/dets.embs")
            return tracks.embs[key], dets.embs[key]
        return tracks.embs, dets.embs

    def _make_pair_tokens(self, track_embs, det_embs):
        B, N, D = track_embs.shape
        M = det_embs.shape[1]

        t = track_embs.unsqueeze(2).expand(B, N, M, D)
        d = det_embs.unsqueeze(1).expand(B, N, M, D)

        tokens = [
            self.pair_cls.view(1, 1, 1, D).expand(B, N, M, D),
            t,
            d,
        ]
        if self.use_absdiff_token:
            tokens.append(torch.abs(t - d))
        if self.use_product_token:
            tokens.append(t * d)

        x = torch.stack(tokens, dim=3)  # [B, N, M, L, D]
        x = x + self.token_type.view(1, 1, 1, self.n_tokens, D)
        return x

    def forward(self, tracks, dets):
        self._requires_attrs(tracks, dets)
        track_embs, det_embs = self._get_embs(tracks, dets)
        B, N, D = track_embs.shape
        M = det_embs.shape[1]

        if N == 0 or M == 0:
            pair_logits = torch.zeros(B, N, M, device=track_embs.device)
            pair_scores = torch.full((B, N, M), -float("inf"), device=track_embs.device)
            tracks.pair_logits = pair_logits
            tracks.pair_scores = pair_scores
            dets.pair_logits = pair_logits
            dets.pair_scores = pair_scores
            self.last_debug = {}
            return pair_scores

        x = self._make_pair_tokens(track_embs, det_embs)  # [B, N, M, L, D]
        x = x.reshape(B * N * M, self.n_tokens, D)

        for layer in self.layers:
            x = layer(x)

        pair_cls_out = x[:, 0, :]
        raw_pair_logits = self.score_head(pair_cls_out).view(B, N, M)
        raw_pair_scores = torch.sigmoid(raw_pair_logits)

        valid_pairs = tracks.masks.unsqueeze(2) & dets.masks.unsqueeze(1)
        pair_scores = raw_pair_scores.masked_fill(~valid_pairs, -float("inf"))
        pair_logits = raw_pair_logits.masked_fill(~valid_pairs, 0.0)

        tracks.pair_logits = pair_logits
        tracks.pair_scores = pair_scores
        dets.pair_logits = pair_logits
        dets.pair_scores = pair_scores

        self.last_debug = {"pair_scores": pair_scores.detach()}
        return pair_scores

# ======================================================================================================================
# ======================================================================================================================
# CP-MLP
# ======================================================================================================================
# ======================================================================================================================
class CP_MLP(BlockB):
    def __init__(
        self,
        cue_dim: int = 1024,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
        expected_num_cues: int = 3,
        use_absdiff: bool = True,
        use_product: bool = True,
        checkpoint_path: Optional[str] = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.cue_dim = cue_dim
        self.expected_num_cues = expected_num_cues
        self.use_absdiff = use_absdiff
        self.use_product = use_product

        flat_dim = expected_num_cues * cue_dim
        in_dim = 2 * flat_dim + int(use_absdiff) * flat_dim + int(use_product) * flat_dim
        self.score_head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.last_debug = {}
        self.init_weights(checkpoint_path=checkpoint_path, module_name="cp_mlp_blockB")

    def _check_inputs(self, tracks, dets):
        required = [(tracks, "cue_ctx"), (dets, "cue_ctx"), (tracks, "masks"), (dets, "masks")]
        missing = [name for obj, name in required if not hasattr(obj, name)]
        if missing:
            raise RuntimeError(f"CP_MLP required inputs. Missing attributes: {missing}")
        if tracks.cue_ctx.dim() != 4 or dets.cue_ctx.dim() != 4:
            raise RuntimeError("CP_MLP expects cue_ctx tensors shaped [B, N/M, K, D]")
        if tracks.cue_ctx.shape[2] != self.expected_num_cues or dets.cue_ctx.shape[2] != self.expected_num_cues:
            raise RuntimeError(
                f"CP_MLP expected {self.expected_num_cues} cues, got "
                f"tracks K={tracks.cue_ctx.shape[2]} and dets K={dets.cue_ctx.shape[2]}"
            )
        if tracks.cue_ctx.shape[-1] != self.cue_dim or dets.cue_ctx.shape[-1] != self.cue_dim:
            raise RuntimeError(
                f"CP_MLP expected cue_dim={self.cue_dim}, got "
                f"tracks D={tracks.cue_ctx.shape[-1]} and dets D={dets.cue_ctx.shape[-1]}"
            )

    def _make_pair_features(self, track_cue_ctx, det_cue_ctx):
        B, N, K, D = track_cue_ctx.shape
        M = det_cue_ctx.shape[1]
        t = track_cue_ctx.reshape(B, N, K * D).unsqueeze(2).expand(B, N, M, K * D)
        d = det_cue_ctx.reshape(B, M, K * D).unsqueeze(1).expand(B, N, M, K * D)
        feats = [t, d]
        if self.use_absdiff:
            feats.append(torch.abs(t - d))
        if self.use_product:
            feats.append(t * d)
        return torch.cat(feats, dim=-1)

    def forward(self, tracks, dets):
        self._check_inputs(tracks, dets)
        track_cue_ctx = tracks.cue_ctx
        det_cue_ctx = dets.cue_ctx
        B, N, K, D = track_cue_ctx.shape
        M = det_cue_ctx.shape[1]
        if N == 0 or M == 0:
            pair_logits = torch.zeros(B, N, M, device=track_cue_ctx.device)
            pair_scores = torch.full((B, N, M), -float("inf"), device=track_cue_ctx.device)
            tracks.pair_logits = dets.pair_logits = pair_logits
            tracks.pair_scores = dets.pair_scores = pair_scores
            self.last_debug = {}
            return pair_scores

        pair_feat = self._make_pair_features(track_cue_ctx, det_cue_ctx)
        raw_pair_logits = self.score_head(pair_feat).squeeze(-1)
        raw_pair_scores = torch.sigmoid(raw_pair_logits)

        valid_pairs = tracks.masks.unsqueeze(2) & dets.masks.unsqueeze(1)
        pair_logits = raw_pair_logits.masked_fill(~valid_pairs, 0.0)
        pair_scores = raw_pair_scores.masked_fill(~valid_pairs, -float("inf"))

        tracks.pair_logits = dets.pair_logits = pair_logits
        tracks.pair_scores = dets.pair_scores = pair_scores
        self.last_debug = {
            "pair_scores": pair_scores.detach(),
            "track_cue_ctx": track_cue_ctx.detach(),
            "det_cue_ctx": det_cue_ctx.detach(),
            "cue_names": getattr(tracks, "cue_names", None),
        }
        return pair_scores


# ======================================================================================================================
# ======================================================================================================================
# CP-T
# ======================================================================================================================
# ======================================================================================================================
class CP_Transformer(BlockB):
    def __init__(
        self,
        cue_dim: int = 1024,
        n_heads: int = 8,
        n_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        expected_num_cues: int = 3,
        use_absdiff_token: bool = True,
        use_product_token: bool = True,
        checkpoint_path: Optional[str] = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.cue_dim = cue_dim
        self.expected_num_cues = expected_num_cues
        self.use_absdiff_token = use_absdiff_token
        self.use_product_token = use_product_token
        self.n_tokens = 1 + 2 * expected_num_cues + int(use_absdiff_token) * expected_num_cues + int(use_product_token) * expected_num_cues

        self.pair_cls = nn.Parameter(torch.zeros(1, 1, cue_dim))
        self.token_type = nn.Parameter(torch.zeros(1, self.n_tokens, cue_dim))
        self.layers = nn.ModuleList([
            _TransformerLayer(cue_dim, n_heads, dim_feedforward, dropout)
            for _ in range(n_layers)
        ])
        self.score_head = nn.Sequential(nn.LayerNorm(cue_dim), nn.Linear(cue_dim, 1))
        self.last_debug = {}
        self.init_weights(checkpoint_path=checkpoint_path, module_name="cp_transformer_blockB")

    def _check_inputs(self, tracks, dets):
        required = [(tracks, "cue_ctx"), (dets, "cue_ctx"), (tracks, "masks"), (dets, "masks")]
        missing = [name for obj, name in required if not hasattr(obj, name)]
        if missing:
            raise RuntimeError(f"CP_Transformer required inputs. Missing attributes: {missing}")
        if tracks.cue_ctx.dim() != 4 or dets.cue_ctx.dim() != 4:
            raise RuntimeError("CP_Transformer expects cue_ctx tensors shaped [B, N/M, K, D]")
        if tracks.cue_ctx.shape[2] != self.expected_num_cues or dets.cue_ctx.shape[2] != self.expected_num_cues:
            raise RuntimeError(
                f"CP_Transformer expected {self.expected_num_cues} cues, got "
                f"tracks K={tracks.cue_ctx.shape[2]} and dets K={dets.cue_ctx.shape[2]}"
            )
        if tracks.cue_ctx.shape[-1] != self.cue_dim or dets.cue_ctx.shape[-1] != self.cue_dim:
            raise RuntimeError(
                f"CP_Transformer expected cue_dim={self.cue_dim}, got "
                f"tracks D={tracks.cue_ctx.shape[-1]} and dets D={dets.cue_ctx.shape[-1]}"
            )

    def _make_pair_tokens(self, track_cue_ctx, det_cue_ctx):
        B, N, K, D = track_cue_ctx.shape
        M = det_cue_ctx.shape[1]
        t = track_cue_ctx.unsqueeze(2).expand(B, N, M, K, D)
        d = det_cue_ctx.unsqueeze(1).expand(B, N, M, K, D)
        tokens = [self.pair_cls.view(1, 1, 1, 1, D).expand(B, N, M, 1, D), t, d]
        if self.use_absdiff_token:
            tokens.append(torch.abs(t - d))
        if self.use_product_token:
            tokens.append(t * d)
        x = torch.cat(tokens, dim=3)  # [B, N, M, L, D]
        x = x + self.token_type.view(1, 1, 1, self.n_tokens, D)
        return x

    def forward(self, tracks, dets):
        self._check_inputs(tracks, dets)
        track_cue_ctx = tracks.cue_ctx
        det_cue_ctx = dets.cue_ctx
        B, N, K, D = track_cue_ctx.shape
        M = det_cue_ctx.shape[1]
        if N == 0 or M == 0:
            pair_logits = torch.zeros(B, N, M, device=track_cue_ctx.device)
            pair_scores = torch.full((B, N, M), -float("inf"), device=track_cue_ctx.device)
            tracks.pair_logits = dets.pair_logits = pair_logits
            tracks.pair_scores = dets.pair_scores = pair_scores
            self.last_debug = {}
            return pair_scores

        x = self._make_pair_tokens(track_cue_ctx, det_cue_ctx).reshape(B * N * M, self.n_tokens, D)
        for layer in self.layers:
            x = layer(x)

        pair_cls_out = x[:, 0, :]
        raw_pair_logits = self.score_head(pair_cls_out).view(B, N, M)
        raw_pair_scores = torch.sigmoid(raw_pair_logits)

        valid_pairs = tracks.masks.unsqueeze(2) & dets.masks.unsqueeze(1)
        pair_logits = raw_pair_logits.masked_fill(~valid_pairs, 0.0)
        pair_scores = raw_pair_scores.masked_fill(~valid_pairs, -float("inf"))

        tracks.pair_logits = dets.pair_logits = pair_logits
        tracks.pair_scores = dets.pair_scores = pair_scores
        self.last_debug = {
            "pair_scores": pair_scores.detach(),
            "pair_cls_out": pair_cls_out.view(B, N, M, D).detach(),
            "cue_names": getattr(tracks, "cue_names", None),
        }
        return pair_scores

# ======================================================================================================================
# ======================================================================================================================
# CA-ES
# ======================================================================================================================
# ======================================================================================================================
class CueAwareEuclideanSimilarity(BlockB):
    """
    Cue-aware similarity scorer.

    Computes:
        sim(i, j) = sum_k sim(cue_k^track_i, cue_k^det_j)
    """

    def __init__(
        self,
        sim_strat: str = "norm_euclidean",
        *args,
        **kwargs,
    ):
        super().__init__()
        if sim_strat not in similarity_metrics:
            raise NotImplementedError(f"Unknown similarity strategy: {sim_strat}")
        self.similarity_metric = similarity_metrics[sim_strat]

    def forward(self, tracks, dets):
        if not hasattr(tracks, "cue_ctx") or not hasattr(dets, "cue_ctx"):
            raise RuntimeError(
                "CueAwareEuclideanSimilarity requires tracks.cue_ctx and dets.cue_ctx"
            )

        track_cues = tracks.cue_ctx  # [B, N, K, D]
        det_cues   = dets.cue_ctx    # [B, M, K, D]

        B, N, K, D = track_cues.shape
        M = det_cues.shape[1]

        sim_per_cue = []
        for k in range(K):
            sim_k = self.similarity_metric(
                track_cues[:, :, k, :], tracks.masks,
                det_cues[:, :, k, :],   dets.masks,
            )  # [B, N, M]
            sim_per_cue.append(sim_k)

        sim_matrix = torch.stack(sim_per_cue, dim=0).sum(dim=0)

        return sim_matrix
        

# ======================================================================================================================
# ======================================================================================================================
# EPACGA
# ======================================================================================================================
# ======================================================================================================================
def _masked_norm_euclidean(track_embs, track_masks, det_embs, det_masks):
    """
    Normalized Euclidean similarity in [0, 1], -inf for invalid pairs.
    """
    track_embs = F.normalize(track_embs, p=2, dim=-1)
    det_embs = F.normalize(det_embs, p=2, dim=-1)

    dist = torch.cdist(track_embs, det_embs, p=2)
    sim = 1.0 - dist / 2.0

    valid = track_masks.unsqueeze(2) & det_masks.unsqueeze(1)
    sim = sim.masked_fill(~valid, -float("inf"))

    return sim


class DAGCA(nn.Module):
    def __init__(
        self,
        emb_dim=1024,
        n_heads=8,
        n_layers=1,
        dim_feedforward=2048,
        gate_hidden_dim=128,
        dropout=0.1,
    ):
        super().__init__()

        self.emb_dim = emb_dim

        self.track_fusion_token = nn.Parameter(torch.zeros(emb_dim))
        self.det_fusion_token = nn.Parameter(torch.zeros(emb_dim))

        self.cue_type_embed = nn.Embedding(4, emb_dim)
        self.role_embed = nn.Embedding(2, emb_dim)

        self.in_norm = nn.LayerNorm(emb_dim)
        self.out_norm = nn.LayerNorm(emb_dim)
        self.in_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.entity_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        self.cue_weight_head = nn.Linear(emb_dim, 3)

        self.gate_net = nn.Sequential(
            nn.Linear(13, gate_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, 3),
            nn.Sigmoid(),
        )

        self.global_weight = nn.Parameter(torch.tensor(1.0))

    def _extract_cues(self, obj):
        """
        Extract cues from RAW cue_ctx
        """
        cues = obj.cue_ctx  # [B, N, K, E]

        assert cues.dim() == 4, f"Expected 4D cue_ctx, got {cues.shape}"
        assert cues.size(2) >= 3, "Need at least 3 cues (app, kp, bbox)"
        assert cues.size(-1) == self.emb_dim, \
            f"Expected embedding dim {self.emb_dim}, got {cues.size(-1)}"

        app  = cues[:, :, 0, :]
        kp   = cues[:, :, 1, :]
        bbox = cues[:, :, 2, :]

        return app, kp, bbox

    @staticmethod
    def _safe(x):
        return torch.where(torch.isfinite(x), x, torch.zeros_like(x))

    def _encode_side(self, obj, is_track):
        app, kp, bbox = self._extract_cues(obj)
        masks = obj.masks

        B, N, E = app.shape

        fusion_token = self.track_fusion_token if is_track else self.det_fusion_token
        fusion = fusion_token.view(1, 1, E).expand(B, N, E)

        bundle = torch.stack([app, kp, bbox, fusion], dim=2)  # [B, N, 4, E]

        cue_ids = torch.tensor([0, 1, 2, 3], device=bundle.device).view(1, 1, 4)
        role_id = 0 if is_track else 1
        role_ids = torch.full((1, 1, 4), role_id, device=bundle.device)

        bundle = bundle + self.cue_type_embed(cue_ids) + self.role_embed(role_ids)

        bundle = self.in_norm(bundle)
        bundle = self.in_drop(bundle)

        bundle = bundle.view(B * N, 4, E)
        encoded = self.entity_encoder(bundle)
        encoded = self.out_norm(encoded)
        encoded = encoded.view(B, N, 4, E)

        cue_tokens = {
            "app": encoded[:, :, 0],
            "kp": encoded[:, :, 1],
            "bbox": encoded[:, :, 2],
        }

        fused = encoded[:, :, 3]

        cue_weights = torch.softmax(self.cue_weight_head(fused), dim=-1)

        if masks is not None:
            fused = fused.masked_fill(~masks.unsqueeze(-1), 0.0)
            cue_weights = cue_weights.masked_fill(~masks.unsqueeze(-1), 0.0)

            for k in cue_tokens:
                cue_tokens[k] = cue_tokens[k].masked_fill(
                    ~masks.unsqueeze(-1), 0.0
                )

        return cue_tokens, fused, cue_weights

    def forward(self, tracks, dets):
        track_cues, track_fused, track_w = self._encode_side(tracks, True)
        det_cues, det_fused, det_w = self._encode_side(dets, False)

        s_app  = _masked_norm_euclidean(track_cues["app"],  tracks.masks, det_cues["app"],  dets.masks)
        s_kp   = _masked_norm_euclidean(track_cues["kp"],   tracks.masks, det_cues["kp"],   dets.masks)
        s_bbox = _masked_norm_euclidean(track_cues["bbox"], tracks.masks, det_cues["bbox"], dets.masks)
        s_glob = _masked_norm_euclidean(track_fused,        tracks.masks, det_fused,        dets.masks)

        cue_sims = torch.stack([s_app, s_kp, s_bbox], dim=-1)

        T = tracks.masks.shape[1]
        D = dets.masks.shape[1]

        gate_in = torch.cat(
            [
                self._safe(s_glob).unsqueeze(-1),
                self._safe(s_app).unsqueeze(-1),
                self._safe(s_kp).unsqueeze(-1),
                self._safe(s_bbox).unsqueeze(-1),

                (self._safe(s_app) - self._safe(s_kp)).abs().unsqueeze(-1),
                (self._safe(s_app) - self._safe(s_bbox)).abs().unsqueeze(-1),
                (self._safe(s_kp) - self._safe(s_bbox)).abs().unsqueeze(-1),

                track_w.unsqueeze(2).expand(-1, -1, D, -1),
                det_w.unsqueeze(1).expand(-1, T, -1, -1),
            ],
            dim=-1,
        )

        gates = self.gate_net(gate_in)

        global_w = F.softplus(self.global_weight) + 1e-6

        numerator = global_w * self._safe(s_glob) + (gates * self._safe(cue_sims)).sum(-1)
        denominator = global_w + gates.sum(-1) + 1e-6

        score_matrix = numerator / denominator

        valid_pairs = tracks.masks.unsqueeze(2) & dets.masks.unsqueeze(1)
        score_matrix = score_matrix.masked_fill(~valid_pairs, -float("inf"))

        tracks.final_pair_score = score_matrix

        return score_matrix
