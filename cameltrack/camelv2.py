import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import transformers
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_metric_learning import distances, losses, reducers

from cameltrack.utils.assignment_strats import association_strats
from cameltrack.utils.coordinates import norm_coords_strats

log = logging.getLogger(__name__)


@dataclass
class Tracklets:
    """Container for tracklets.

    features: dict of tensors float32 [B, N, T, F]
    feats_masks: tensor bool [B, N, T]
    targets: optional tensor [B, N] or [B, N, T]
    """
    def __init__(self, features, feats_masks, targets=None):
        self.feats = features
        self.feats_masks = feats_masks
        self.masks = self.feats_masks.any(dim=-1)
        if targets is not None and len(targets.shape) > 2:
            self.targets = targets[:, :, 0]
        else:
            self.targets = targets


@dataclass
class Detections(Tracklets):
    """Container for detections.

    features: dict of tensors float32 [B, N, 1, F]
    feats_masks: tensor bool [B, N, 1]
    targets: optional tensor [B, N] or [B, N, 1]
    """
    def __init__(self, features, feats_masks, targets=None):
        assert feats_masks.shape[2] == 1
        super().__init__(features, feats_masks, targets)


class CAMELV2(pl.LightningModule):
    """CAMEL with explicit two-block association architecture.

    Pipeline:
        (i)   tokenize raw track/detection features into cue tokens
        (ii)  blockA: encode tokens into discriminative track/detection embeddings
        (iii) blockB: score track/detection pairs and return association scores
        (iv)  association matrix = blockB output
        (v)   Hungarian/association strategy is applied in predict_step

    blockA contract:
        tracks, dets = blockA(tracks, dets)
        must set tracks.embs and dets.embs, either tensors [B,N,D]/[B,M,D]
        or dictionaries of tensors for multi-token/multi-cue embeddings.

    blockB contract:
        td_sim_matrix = blockB(tracks, dets)
        must return [B,N,M] scores where larger means more likely association.
        It may also set tracks.pair_logits / tracks.pair_scores for pairwise losses.
    """

    def __init__(
        self,
        blockA: DictConfig,
        blockB: DictConfig,
        preprocessing: DictConfig,
        temporal_encoders: DictConfig,
        sim_threshold: float = 0.5,
        use_computed_sim_threshold: bool = False,
        optimizer: DictConfig = None,
        ass_strat: str = "hungarian_algorithm",
        norm_strat: str = "positive",
        use_pair_loss: bool = True,
        pair_loss_weight: float = 0.5,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=[key for key in locals() if "checkpoint_path" in key])

        self.preprocessing = instantiate(preprocessing, _recursive_=False)
        self.blockA = instantiate(blockA, _recursive_=False)
        self.blockB = instantiate(blockB, _recursive_=False)

        cue_dim = getattr(self.blockA, "cue_dim", None)
        if cue_dim is None:
            cue_dim = getattr(self.blockB, "cue_dim", 1024)
            log.warning(
                "blockA has no cue_dim attribute. Falling back to cue_dim=%s for temporal encoders.", cue_dim
            )

        self.temp_encs = nn.ModuleDict({
            name: instantiate(enc, output_dim=cue_dim, name=name, _recursive_=False)
            for name, enc in temporal_encoders.items()
        })

        self.sim_threshold = sim_threshold
        self.use_computed_sim_threshold = use_computed_sim_threshold
        self.computed_sim_threshold = None
        if use_computed_sim_threshold:
            log.warning(
                "CAMELV2 initialized with sim_threshold=%s. This may be updated later by validation.",
                sim_threshold,
            )

        self.use_pair_loss = use_pair_loss
        self.pair_loss_weight = pair_loss_weight

        if ass_strat in association_strats:
            self.association = association_strats[ass_strat]
        else:
            raise NotImplementedError(f"Unknown association strategy: {ass_strat}")

        if norm_strat in norm_coords_strats:
            self.norm_coords = norm_coords_strats[norm_strat]
        else:
            raise NotImplementedError(f"Unknown coordinate normalization strategy: {norm_strat}")

        self.optimizer = optimizer
        self.sim_loss = losses.NTXentLoss(
            distance=distances.CosineSimilarity(),
            reducer=reducers.AvgNonZeroReducer(),
        )
        self.counter = 0

    def training_step(self, batch, batch_idx):
        tracks, dets = self.train_val_preprocess(batch)
        tracks, dets, td_sim_matrix = self.forward(tracks, dets)
        loss = self.compute_loss(tracks, dets, td_sim_matrix)
        self.log_loss(loss, "train")
        return {"loss": loss, "dets": dets, "tracks": tracks, "td_sim_matrix": td_sim_matrix}

    def validation_step(self, batch, batch_idx):
        tracks, dets = self.train_val_preprocess(batch)
        tracks, dets, td_sim_matrix = self.forward(tracks, dets)
        loss = self.compute_loss(tracks, dets, td_sim_matrix)
        self.log_loss(loss, "val")
        return {"loss": loss, "tracks": tracks, "dets": dets, "td_sim_matrix": td_sim_matrix}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        tracks, dets = self.predict_preprocess(batch)
        tracks, dets, td_sim_matrix = self.forward(tracks, dets)
        sim_threshold = (
            self.computed_sim_threshold
            if (self.use_computed_sim_threshold and self.computed_sim_threshold)
            else self.sim_threshold
        )
        association_matrix, association_result = self.association(
            td_sim_matrix, tracks.masks, dets.masks, sim_threshold=sim_threshold
        )
        return association_matrix, association_result, td_sim_matrix

    def forward(self, tracks, dets):
        self.counter += 1

        # 1) Tokenize raw features into cue tokens.
        tracks, dets = self.tokenize(tracks, dets)

        # 2) Preprocess cue tokens into tracks.cue_ctx / dets.cue_ctx.
        tracks, dets = self.preprocessing(tracks, dets)

        # 3) Block A: discriminative entity representation.
        tracks, dets = self.blockA(tracks, dets)
        if not hasattr(tracks, "embs") or not hasattr(dets, "embs"):
            raise RuntimeError("blockA must set tracks.embs and dets.embs")

        # 4) Block B: Pairwise association scoring.
        td_sim_matrix = self.blockB(tracks, dets)
        if td_sim_matrix is None:
            raise RuntimeError("blockB must return a [B,N,M] association score matrix")

        return tracks, dets, td_sim_matrix

    def train_val_preprocess(self, batch):
        if self.norm_coords is not None:
            batch = self.norm_coords(batch)
        tracks = Tracklets(batch["track_feats"], ~batch["track_targets"].isnan(), batch["track_targets"])
        dets = Detections(batch["det_feats"], ~batch["det_targets"].isnan(), batch["det_targets"])
        return tracks, dets

    def predict_preprocess(self, batch):
        if self.norm_coords is not None:
            batch = self.norm_coords(batch)
        tracks = Tracklets(
            batch["track_feats"],
            batch["track_masks"],
            batch["track_targets"] if "track_targets" in batch else None,
        )
        dets = Detections(
            batch["det_feats"],
            batch["det_masks"],
            batch["det_targets"] if "det_targets" in batch else None,
        )
        return tracks, dets

    def tokenize(self, tracks, dets):
        tracks.tokens = {}
        dets.tokens = {}
        for name, temp_enc in self.temp_encs.items():
            tracks.tokens[name] = temp_enc(tracks)
            dets.tokens[name] = temp_enc(dets)
        return tracks, dets

    def compute_loss(self, tracks, dets, td_sim_matrix=None):
        """Original embedding loss plus optional pairwise BCE from blockB."""
        if not isinstance(tracks.embs, dict):
            tracks.embs = {"default": tracks.embs}
            dets.embs = {"default": dets.embs}

        n_tokens = len(tracks.embs.keys())
        B = list(tracks.embs.values())[0].shape[0]
        sim_loss = torch.zeros((n_tokens, B), dtype=torch.float32, device=self.device)
        mask_sim_loss = torch.zeros((n_tokens, B), dtype=torch.bool, device=self.device)

        for h, token_name in enumerate(tracks.embs.keys()):
            tracks_embs = tracks.embs[token_name]
            dets_embs = dets.embs[token_name]
            for i in range(B):
                masked_track_embs = tracks_embs[i, tracks.masks[i]]
                masked_track_targets = tracks.targets[i, tracks.masks[i]]
                masked_det_embs = dets_embs[i, dets.masks[i]]
                masked_det_targets = dets.targets[i, dets.masks[i]]

                if ((len(masked_det_embs) != 0 or len(masked_track_embs) != 0) or
                    (len(masked_det_embs) > 1 or len(masked_track_embs) > 1)):
                    mask_sim_loss[h, i] = True

                    valid_tracks = masked_track_targets >= 0
                    valid_dets = masked_det_targets >= 0
                    embeddings = torch.cat([
                        masked_track_embs[valid_tracks],
                        masked_det_embs[valid_dets],
                    ], dim=0)
                    labels = torch.cat([
                        masked_track_targets[valid_tracks],
                        masked_det_targets[valid_dets],
                    ], dim=0)
                    sim_loss[h, i] = self.sim_loss(embeddings, labels)

        sim_loss = sim_loss[mask_sim_loss].mean().nan_to_num(0)

        pair_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        if (
            self.use_pair_loss
            and hasattr(tracks, "pair_logits")
            and tracks.targets is not None
            and dets.targets is not None
        ):
            valid_pairs = tracks.masks.unsqueeze(2) & dets.masks.unsqueeze(1)
            valid_pairs = valid_pairs & (tracks.targets.unsqueeze(2) >= 0) & (dets.targets.unsqueeze(1) >= 0)
            if valid_pairs.any():
                pair_targets = (tracks.targets.unsqueeze(2) == dets.targets.unsqueeze(1)).float()
                logits = tracks.pair_logits[valid_pairs]
                targets = pair_targets[valid_pairs]
                n_pos = targets.sum().item()
                n_all = targets.numel()
                n_neg = n_all - n_pos
                if n_pos > 0:
                    pos_weight = torch.tensor([max(n_neg / n_pos, 1.0)], device=self.device)
                    pair_loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
                else:
                    pair_loss = F.binary_cross_entropy_with_logits(logits, targets)

        return sim_loss + self.pair_loss_weight * pair_loss

    def log_loss(self, loss, step):
        self.log_dict(
            {f"{step}/loss": loss},
            on_epoch=True,
            on_step="train" == step,
            prog_bar="train" == step,
            logger=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.optimizer.init_lr, weight_decay=self.optimizer.weight_decay
        )
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.trainer.estimated_stepping_batches // 20,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def on_save_checkpoint(self, checkpoint):
        checkpoint["computed_sim_threshold"] = self.computed_sim_threshold

    def on_load_checkpoint(self, checkpoint):
        self.computed_sim_threshold = checkpoint.get("computed_sim_threshold", None)
