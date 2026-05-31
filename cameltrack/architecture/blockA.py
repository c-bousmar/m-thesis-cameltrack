import logging
import math
from typing import Sequence, Optional

import torch
import torch.nn as nn

from cameltrack.architecture.base_module import Module
from cameltrack.architecture.gaffe import GAFFE as _GAFFE
from cameltrack.utils.merge_token_strats import merge_token_strats

log = logging.getLogger(__name__)


class BlockA(Module):
    """Base interface for representation blocks.

    Contract:
        tracks, dets = blockA(tracks, dets)
        tracks.embs, dets.embs, tracks.cue_ctx, dets.cue_ctx must be set.
    """
    cue_dim: int = 1024

    def forward(self, tracks, dets):
        raise NotImplementedError

# ======================================================================================================================
# ======================================================================================================================
# Mock BlockA, passthrough from preprocessing to blockB
# ======================================================================================================================
# ======================================================================================================================
class Mock(nn.Module):
    def __init__(self, merge_token_strat: str = "sum"):
        super().__init__()

        self.merge_token_strat = merge_token_strat

        if merge_token_strat not in merge_token_strats:
            raise NotImplementedError(
                f"Unknown merge_token_strat={merge_token_strat}. "
                f"Available: {list(merge_token_strats.keys())}"
            )
        self.merge = merge_token_strats[merge_token_strat]

    def forward(self, tracks, dets):
        tracks, dets = self.merge(tracks, dets)
        tracks.embs = tracks.tokens
        dets.embs = dets.tokens
        return tracks, dets
        
# ======================================================================================================================
# ======================================================================================================================
# Original GAFFE
# ======================================================================================================================
# ======================================================================================================================
class GAFFE(_GAFFE):
    """Adapter for running original CAMELTrack GAFFE inside the BlockA API.

    This adapter applies the original CAMEL merge strategy before
    calling the original GAFFE forward().
    """
    def __init__(
        self,
        emb_dim: int = 1024,
        n_heads: int = 8,
        n_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        merge_token_strat: str = "sum",
        checkpoint_path: str = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            emb_dim=emb_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            checkpoint_path=checkpoint_path,
            *args,
            **kwargs,
        )

        self.emb_dim = emb_dim
        self.cue_dim = emb_dim
        self.merge_token_strat = merge_token_strat

        if merge_token_strat not in merge_token_strats:
            raise NotImplementedError(
                f"Unknown merge_token_strat={merge_token_strat}. "
                f"Available: {list(merge_token_strats.keys())}"
            )
        self.merge = merge_token_strats[merge_token_strat]

    def forward(self, tracks, dets):
        tracks, dets = self.merge(tracks, dets)
        return super().forward(tracks, dets)

# ======================================================================================================================
# ======================================================================================================================
# IIEA
# ======================================================================================================================
# ======================================================================================================================
class _SelfAttnBlock(nn.Module):
    """Small private helper used by IIABlockA."""

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

    def forward(self, x, key_padding_mask=None):
        attn_out, attn_weights = self.attn(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x, attn_weights


class IIA(BlockA):
    """IIA: intra-entity and inter-entity transformers.

    It exposes the intermediate tensors:
        tracks.cls_ctx:  [B, N, D]
        dets.cls_ctx:    [B, M, D]
        tracks.cue_ctx: [B, N, K, D]
        dets.cue_ctx:   [B, M, K, D]
        tracks.cues_raw: [B, N, K, D]
        dets.cues_raw:   [B, M, K, D]
        tracks.cue_names / dets.cue_names
    """

    def __init__(
        self,
        cue_dim: int = 1024,
        cue_names: Sequence[str] = ("app_encoder", "kp_encoder", "bbox_encoder"),
        n_heads: int = 8,
        intra_layers: int = 1,
        global_layers: int = 2,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        shuffle_entities_during_train: bool = True,
        checkpoint_path: Optional[str] = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.cue_dim = cue_dim
        self.cue_names = tuple(cue_names)
        self.num_cues = len(self.cue_names)
        self.shuffle_entities_during_train = shuffle_entities_during_train

        self.track_cls = nn.Parameter(torch.zeros(1, 1, 1, cue_dim))
        self.det_cls = nn.Parameter(torch.zeros(1, 1, 1, cue_dim))

        self.local_token_type = nn.Parameter(torch.zeros(1, 1, 1 + self.num_cues, cue_dim))

        self.local_track_side = nn.Parameter(torch.zeros(1, 1, 1, cue_dim))
        self.local_det_side = nn.Parameter(torch.zeros(1, 1, 1, cue_dim))
        self.global_track_side = nn.Parameter(torch.zeros(1, 1, cue_dim))
        self.global_det_side = nn.Parameter(torch.zeros(1, 1, cue_dim))

        self.intra_blocks = nn.ModuleList([
            _SelfAttnBlock(cue_dim, n_heads, dim_feedforward, dropout)
            for _ in range(intra_layers)
        ])
        self.global_blocks = nn.ModuleList([
            _SelfAttnBlock(cue_dim, n_heads, dim_feedforward, dropout)
            for _ in range(global_layers)
        ])

        self.track_out = nn.Sequential(nn.LayerNorm(cue_dim), nn.Linear(cue_dim, cue_dim))
        self.det_out = nn.Sequential(nn.LayerNorm(cue_dim), nn.Linear(cue_dim, cue_dim))

        self.last_debug = {}
        self.init_weights(checkpoint_path=checkpoint_path, module_name="IIA")

    def _stack_cues(self, token_dict):
        missing = [name for name in self.cue_names if name not in token_dict]
        if missing:
            raise KeyError(f"Missing cues in token dict: {missing}. Found: {list(token_dict.keys())}")
        return torch.stack([token_dict[name] for name in self.cue_names], dim=2)  # [B, E, K, D]

    def _encode_entities(self, cues, entity_mask, is_track: bool):
        B, E, K, D = cues.shape
        device = cues.device

        cls_token = self.track_cls if is_track else self.det_cls
        side_embed = self.local_track_side if is_track else self.local_det_side

        cls = cls_token.expand(B, E, 1, D)
        x = torch.cat([cls, cues], dim=2)  # [B, E, 1+K, D]
        x = x + self.local_token_type[:, :, : 1 + K, :] + side_embed

        token_mask = torch.cat(
            [
                torch.ones(B, E, 1, dtype=torch.bool, device=device),
                entity_mask.unsqueeze(-1).expand(B, E, K),
            ],
            dim=2,
        )

        x = x.view(B * E, 1 + K, D)
        key_padding_mask = ~token_mask.view(B * E, 1 + K)

        for blk in self.intra_blocks:
            x, _ = blk(x, key_padding_mask=key_padding_mask)

        x = x.view(B, E, 1 + K, D)
        x = x * token_mask.unsqueeze(-1)

        cls = x[:, :, 0, :]
        cue_ctx = x[:, :, 1:, :]
        return cls, cue_ctx

    def _global_context(self, track_cls, det_cls, track_mask, det_mask):
        B, N, D = track_cls.shape
        M = det_cls.shape[1]
        L = N + M
        device = track_cls.device

        x = torch.cat(
            [
                track_cls + self.global_track_side,
                det_cls + self.global_det_side,
            ],
            dim=1,
        )  # [B, N+M, D]
        mask = torch.cat([track_mask, det_mask], dim=1)  # [B, N+M]

        inv_perm = None
        if self.training and self.shuffle_entities_during_train and L > 1:
            perm = torch.randperm(L, device=device)
            inv_perm = torch.argsort(perm)
            x = x[:, perm, :]
            mask = mask[:, perm]

        for blk in self.global_blocks:
            x, _ = blk(x, key_padding_mask=~mask)

        if inv_perm is not None:
            x = x[:, inv_perm, :]

        return x[:, :N, :], x[:, N:, :]

    def forward(self, tracks, dets):
        device = next(self.parameters()).device
        B = next(iter(tracks.tokens.values())).shape[0]
        N = next(iter(tracks.tokens.values())).shape[1]
        M = next(iter(dets.tokens.values())).shape[1]

        if N == 0 or M == 0:
            tracks.embs = torch.zeros(B, N, self.cue_dim, device=device)
            dets.embs = torch.zeros(B, M, self.cue_dim, device=device)

            tracks.cls_ctx = torch.zeros(B, N, self.cue_dim, device=device)
            dets.cls_ctx = torch.zeros(B, M, self.cue_dim, device=device)
            tracks.cue_ctx = torch.zeros(B, N, self.num_cues, self.cue_dim, device=device)
            dets.cue_ctx = torch.zeros(B, M, self.num_cues, self.cue_dim, device=device)
            tracks.cues_raw = tracks.cue_ctx
            dets.cues_raw = dets.cue_ctx
            tracks.cue_names = self.cue_names
            dets.cue_names = self.cue_names
            self.last_debug = {}
            return tracks, dets

        track_mask = tracks.masks
        det_mask = dets.masks

        track_cues = self._stack_cues(tracks.tokens)  # [B, N, K, D]
        det_cues = self._stack_cues(dets.tokens)      # [B, M, K, D]

        track_cls_local, track_cue_ctx = self._encode_entities(track_cues, track_mask, is_track=True)
        det_cls_local, det_cue_ctx = self._encode_entities(det_cues, det_mask, is_track=False)

        track_cls_ctx, det_cls_ctx = self._global_context(
            track_cls_local, det_cls_local, track_mask, det_mask
        )

        tracks.embs = track_cls_ctx
        dets.embs = det_cls_ctx

        tracks.cls_ctx = track_cls_ctx
        dets.cls_ctx = det_cls_ctx
        tracks.cue_ctx = track_cue_ctx
        dets.cue_ctx = det_cue_ctx
        tracks.cues_raw = track_cues
        dets.cues_raw = det_cues
        tracks.cue_names = self.cue_names
        dets.cue_names = self.cue_names

        self.last_debug = {
            "track_cls_ctx": track_cls_ctx.detach(),
            "det_cls_ctx": det_cls_ctx.detach(),
            "track_cue_ctx": track_cue_ctx.detach(),
            "det_cue_ctx": det_cue_ctx.detach(),
            "cue_names": self.cue_names,
        }
        return tracks, dets


# ======================================================================================================================
# ======================================================================================================================
# MCA
# ======================================================================================================================
# ======================================================================================================================
class MCA(BlockA):
    """
    It produces only:
        tracks.embs: [B, N, D]
        dets.embs:   [B, M, D]
    """

    def __init__(
        self,
        emb_dim: int = 1024,
        cue_names: Sequence[str] = ("app_encoder", "bbox_encoder", "kp_encoder"),
        n_heads: int = 8,
        n_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        normalize_output: bool = True,
        checkpoint_path: Optional[str] = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.cue_dim = emb_dim
        self.cue_names = tuple(cue_names)
        self.num_cues = len(self.cue_names)
        self.normalize_output = normalize_output

        self.modal_types = nn.ParameterDict({
            name: nn.Parameter(torch.randn(emb_dim) * 0.02)
            for name in self.cue_names
        })

        self.track_source_emb = nn.Parameter(torch.randn(emb_dim) * 0.02)
        self.det_source_emb = nn.Parameter(torch.randn(emb_dim) * 0.02)

        self.fusion_proj = nn.Linear(emb_dim * self.num_cues, emb_dim)
        self.norm = nn.LayerNorm(emb_dim)
        self.drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.init_weights(checkpoint_path=checkpoint_path, module_name="mca_blockA")

    def _build_modal_tokens(self, token_dict, source_emb):
        missing = [name for name in self.cue_names if name not in token_dict]
        if missing:
            raise KeyError(f"Missing cue tokens: {missing}. Found: {list(token_dict.keys())}")

        tokens = []
        for name in self.cue_names:
            tokens.append(token_dict[name] + self.modal_types[name] + source_emb)
        return tokens

    def forward(self, tracks, dets):
        track_modal_tokens = self._build_modal_tokens(tracks.tokens, self.track_source_emb)
        det_modal_tokens = self._build_modal_tokens(dets.tokens, self.det_source_emb)

        B, N, D = track_modal_tokens[0].shape
        M = det_modal_tokens[0].shape[1]

        cls_t = self.fusion_proj(torch.cat(track_modal_tokens, dim=-1))  # [B, N, D]
        cls_d = self.fusion_proj(torch.cat(det_modal_tokens, dim=-1))    # [B, M, D]
        cls_entities = torch.cat([cls_t, cls_d], dim=1)                  # [B, N+M, D]

        modal_tokens = torch.cat(track_modal_tokens + det_modal_tokens, dim=1)
        src = torch.cat([cls_entities, modal_tokens], dim=1)
        src = self.drop(self.norm(src))

        cls_mask = torch.cat([tracks.masks, dets.masks], dim=1)  # [B, N+M]
        track_modal_mask = torch.cat([tracks.masks for _ in self.cue_names], dim=1)  # [B, K*N]
        det_modal_mask = torch.cat([dets.masks for _ in self.cue_names], dim=1)      # [B, K*M]
        full_mask = torch.cat([cls_mask, track_modal_mask, det_modal_mask], dim=1)

        x = self.transformer(src, src_key_padding_mask=~full_mask)

        tracks_out = x[:, :N, :]
        dets_out = x[:, N:N + M, :]

        if self.normalize_output:
            tracks_out = torch.nn.functional.normalize(tracks_out, p=2, dim=-1)
            dets_out = torch.nn.functional.normalize(dets_out, p=2, dim=-1)

        tracks.embs = tracks_out
        dets.embs = dets_out
        tracks.cue_ctx = torch.stack(track_modal_tokens, dim=2)  # [B, N, K, D]
        dets.cue_ctx   = torch.stack(det_modal_tokens,   dim=2)  # [B, M, K, D]
        return tracks, dets

# ======================================================================================================================
# ======================================================================================================================
# ECCA
# ======================================================================================================================
# ======================================================================================================================
class _ECCA_Block(nn.Module):
    """Small private helper used by ECCA."""

    def __init__(self, dim: int, n_heads: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim)
        )
        self.norm3 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, memory, mem_mask=None):
        q2 = self.norm1(queries)
        self_output, _ = self.self_attn(q2, q2, q2, need_weights=False)
        queries = queries + self.dropout(self_output)
        
        q2 = self.norm2(queries)
        attn_output, _ = self.cross_attn(
            q2, memory, memory, 
            key_padding_mask=~mem_mask if mem_mask is not None else None,
            need_weights=False 
        )
        queries = queries + self.dropout(attn_output)
        
        queries = queries + self.dropout(self.mlp(self.norm3(queries)))
        return queries


class ECCA(BlockA):
    """ECCA: Entity-Cue Cross-Attention mechanism for cross-modal association."""
    def __init__(
        self,
        cue_dim: int = 1024,
        n_heads: int = 8,
        num_blocks: int = 2,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        checkpoint_path: Optional[str] = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.cue_dim = cue_dim
        self.blocks = nn.ModuleList([
            _ECCA_Block(cue_dim, n_heads, dim_feedforward, dropout) 
            for _ in range(num_blocks)
        ])
        self.init_weights(checkpoint_path=checkpoint_path, module_name="ecca_blockA")

    def forward(self, tracks, dets):
        device = next(self.parameters()).device
        
        B = next(iter(tracks.tokens.values())).shape[0]
        N = next(iter(tracks.tokens.values())).shape[1]
        M = next(iter(dets.tokens.values())).shape[1]

        if N == 0 or M == 0:
            tracks.embs = torch.zeros(B, N, self.cue_dim, device=device)
            dets.embs = torch.zeros(B, M, self.cue_dim, device=device)
            return tracks, dets

        all_cues = []
        all_masks = []
        for entity_set in [tracks, dets]:
            for cue_name, cue_tensor in entity_set.tokens.items():
                all_cues.append(cue_tensor)
                all_masks.append(entity_set.masks)
        
        memory = torch.cat(all_cues, dim=1)    
        mem_mask = torch.cat(all_masks, dim=1) 

        track_init = torch.stack(list(tracks.tokens.values())).sum(dim=0)
        det_init = torch.stack(list(dets.tokens.values())).sum(dim=0)
        queries = torch.cat([track_init, det_init], dim=1) 

        for block in self.blocks:
            queries = block(queries, memory, mem_mask=mem_mask)

        tracks.embs = queries[:, :N, :]
        dets.embs = queries[:, N:, :]

        return tracks, dets
