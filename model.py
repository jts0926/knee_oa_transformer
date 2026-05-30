from typing import Dict, List

import torch
import torch.nn as nn
import math
import warnings
from torchvision.models import ResNet18_Weights, resnet18

from config import (
    ROLE_TO_INDEX,
    TIMEPOINT_TO_INDEX,
    TOKEN_METADATA,
    VIEW_TO_INDEX,
    Config,
)


class ResNet18ImageEncoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if cfg.model.pretrained_resnet18 else None
        try:
            backbone = resnet18(weights=weights)
        except Exception as exc:
            if weights is None:
                raise
            warnings.warn(
                "Could not load/download ImageNet-pretrained ResNet-18 weights. "
                "Falling back to random initialization. To avoid this, cache the "
                "weights before running on an offline machine or set "
                "cfg.model.pretrained_resnet18 = False.",
                RuntimeWarning,
            )
            backbone = resnet18(weights=None)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cfg.model.projection_in_dim, cfg.model.embedding_dim),
            nn.LayerNorm(cfg.model.embedding_dim),
            nn.GELU(),
            nn.Dropout(cfg.model.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(x))


class KneeOAClassificationTransformer(nn.Module):
    def __init__(self, cfg: Config, model_name: str):
        super().__init__()
        self.cfg = cfg
        self.model_name = model_name
        self.architecture = str(cfg.model_inputs[model_name].get("architecture", "global_transformer"))
        self.tokens: List[str] = self._resolve_tokens(list(cfg.model_inputs[model_name]["tokens"]))
        self.required_images: List[str] = list(cfg.model_inputs[model_name]["required_images"])
        dim = cfg.model.embedding_dim

        self.image_encoder = ResNet18ImageEncoder(cfg)
        self._last_attention_weights: Dict[str, Dict[str, float]] = {}
        if self.architecture == "patch_mil":
            self._init_patch_mil(dim)
            return

        self.view_embedding = nn.Embedding(len(VIEW_TO_INDEX), dim)
        self.timepoint_embedding = nn.Embedding(len(TIMEPOINT_TO_INDEX), dim)
        self.role_embedding = nn.Embedding(len(ROLE_TO_INDEX), dim)
        self.time_gap_projection = nn.Sequential(
            nn.Linear(1, dim),
            nn.Tanh(),
            nn.Linear(dim, dim),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.token_gate_logits = nn.ParameterDict(
            {
                token: nn.Parameter(torch.tensor(self._initial_gate_logit(token), dtype=torch.float32))
                for token in self.tokens
            }
        )
        self.relation_projectors = nn.ModuleDict(
            {
                token: nn.Sequential(
                    nn.Linear(dim * 4, dim),
                    nn.LayerNorm(dim),
                    nn.GELU(),
                    nn.Dropout(cfg.model.dropout),
                    nn.Linear(dim, dim),
                )
                for token in self.tokens
                if token.startswith("relation_")
            }
        )
        self.pool_score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1),
        )
        self.task_pool_score = nn.ModuleDict(
            {
                "incidence": nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1)),
                "progression": nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1)),
            }
        )
        self.context_scale_logits = nn.ParameterDict(
            {
                task: nn.Parameter(torch.tensor(self._context_scale_logit(), dtype=torch.float32))
                for task in ["incidence", "progression"]
            }
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=cfg.model.attention_heads,
            dim_feedforward=cfg.model.feedforward_dim,
            dropout=cfg.model.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.model.transformer_layers)
        self.norm = nn.LayerNorm(dim)
        self.incidence_head = nn.Sequential(
            nn.Linear(dim, 128),
            nn.GELU(),
            nn.Dropout(cfg.model.dropout),
            nn.Linear(128, 1),
        )
        self.progression_head = nn.Sequential(
            nn.Linear(dim, 128),
            nn.GELU(),
            nn.Dropout(cfg.model.dropout),
            nn.Linear(128, 1),
        )

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.cfg.model.fusion_type not in {"anchor_attention", "task_attention", "transformer", "gated_pooling"}:
            raise ValueError(
                "cfg.model.fusion_type must be 'anchor_attention', 'task_attention', 'transformer', or 'gated_pooling'"
            )

    def _init_patch_mil(self, dim: int) -> None:
        if len(self.required_images) != 1:
            raise ValueError("patch_mil currently expects exactly one required image.")
        grid = int(self.cfg.training.patch_grid_size)
        if self.cfg.training.patch_image_size % grid != 0:
            raise ValueError("patch_image_size must be divisible by patch_grid_size.")
        self.patch_grid_size = grid
        self.patch_size = int(self.cfg.training.patch_image_size // grid)
        self.num_patches = grid * grid
        self.patch_pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=self.cfg.model.attention_heads,
            dim_feedforward=self.cfg.model.feedforward_dim,
            dropout=self.cfg.model.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.patch_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.cfg.model.patch_transformer_layers,
        )
        self.norm = nn.LayerNorm(dim)
        self.task_pool_score = nn.ModuleDict(
            {
                "incidence": nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1)),
                "progression": nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1)),
            }
        )
        self.incidence_head = nn.Sequential(
            nn.Linear(dim, 128),
            nn.GELU(),
            nn.Dropout(self.cfg.model.dropout),
            nn.Linear(128, 1),
        )
        self.progression_head = nn.Sequential(
            nn.Linear(dim, 128),
            nn.GELU(),
            nn.Dropout(self.cfg.model.dropout),
            nn.Linear(128, 1),
        )
        self.token_gate_logits = nn.ParameterDict()
        self.context_scale_logits = nn.ParameterDict()
        nn.init.trunc_normal_(self.patch_pos_embedding, std=0.02)

    def _resolve_tokens(self, tokens: List[str]) -> List[str]:
        model_uses_delta = bool(self.cfg.model_inputs[self.model_name].get("use_delta_tokens", False))
        if self.cfg.model.use_delta_tokens or model_uses_delta:
            return tokens
        return [token for token in tokens if not token.startswith("delta_")]

    def _initial_gate_value(self, token: str) -> float:
        if not self.cfg.model.use_clinical_token_gates:
            return 1.0
        if token == "delta_pa":
            return self.cfg.model.initial_delta_pa_gate
        if token == "delta_lat":
            return self.cfg.model.initial_delta_lat_gate
        meta = TOKEN_METADATA[token]
        if meta["view"] == "pa":
            return self.cfg.model.initial_pa_gate
        return self.cfg.model.initial_lat_gate

    def _initial_gate_logit(self, token: str) -> float:
        value = min(max(self._initial_gate_value(token), 1e-4), 1.0 - 1e-4)
        return math.log(value / (1.0 - value))

    def _context_scale_logit(self) -> float:
        value = min(max(self.cfg.model.anchor_context_init, 1e-4), 1.0 - 1e-4)
        return math.log(value / (1.0 - value))

    def _token_gate(self, token: str) -> torch.Tensor:
        if not self.cfg.model.use_clinical_token_gates:
            return torch.ones((), device=self.token_gate_logits[token].device)
        return torch.sigmoid(self.token_gate_logits[token])

    def token_gate_values(self) -> Dict[str, float]:
        if len(self.token_gate_logits) == 0:
            return {}
        return {
            token: float(torch.sigmoid(parameter).detach().cpu())
            for token, parameter in self.token_gate_logits.items()
        }

    def token_attention_values(self) -> Dict[str, Dict[str, float]]:
        return self._last_attention_weights

    def context_scale_values(self) -> Dict[str, float]:
        if len(self.context_scale_logits) == 0:
            return {}
        return {
            task: float(torch.sigmoid(parameter).detach().cpu())
            for task, parameter in self.context_scale_logits.items()
        }

    def _metadata_embedding(self, token: str, batch_size: int, device: torch.device) -> torch.Tensor:
        meta = TOKEN_METADATA[token]
        view = torch.full((batch_size,), VIEW_TO_INDEX[meta["view"]], dtype=torch.long, device=device)
        timepoint = torch.full((batch_size,), TIMEPOINT_TO_INDEX[meta["timepoint"]], dtype=torch.long, device=device)
        role = torch.full((batch_size,), ROLE_TO_INDEX[meta["role"]], dtype=torch.long, device=device)
        embedding = self.view_embedding(view) + self.timepoint_embedding(timepoint) + self.role_embedding(role)
        if self.cfg.model.use_continuous_time_gap_embedding:
            embedding = embedding + self._time_gap_embedding(token, batch_size, device)
        return embedding * float(self.cfg.model.metadata_scale)

    def _time_gap_embedding(self, token: str, batch_size: int, device: torch.device) -> torch.Tensor:
        meta = TOKEN_METADATA[token]
        if meta["timepoint"] == "baseline":
            months_from_landmark = -self.cfg.model.baseline_months_before_landmark
        elif meta["role"] in {"change", "relation"}:
            months_from_landmark = -self.cfg.model.baseline_months_before_landmark
        else:
            months_from_landmark = 0.0
        scaled = months_from_landmark / max(self.cfg.model.baseline_months_before_landmark, 1.0)
        values = torch.full((batch_size, 1), scaled, dtype=torch.float32, device=device)
        return self.time_gap_projection(values)

    def _build_base_embeddings(self, images: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        embeddings = {}
        for image_key in self.required_images:
            embeddings[image_key] = self.image_encoder(images[image_key])
        if "delta_pa" in self.tokens:
            embeddings["delta_pa"] = embeddings["m30_pa"] - embeddings["bl_pa"]
        if "delta_lat" in self.tokens:
            embeddings["delta_lat"] = embeddings["m30_lat"] - embeddings["bl_lat"]
        if "relation_pa" in self.tokens:
            embeddings["relation_pa"] = self._relation_embedding("relation_pa", embeddings["m30_pa"], embeddings["bl_pa"])
        if "relation_lat" in self.tokens:
            embeddings["relation_lat"] = self._relation_embedding("relation_lat", embeddings["m30_lat"], embeddings["bl_lat"])
        return embeddings

    def _relation_embedding(self, token: str, current: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
        diff = current - baseline
        relation = torch.cat([current, baseline, diff, diff.abs()], dim=1)
        return self.relation_projectors[token](relation)

    def _heads(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "incidence": self.incidence_head(features).reshape(-1),
            "progression": self.progression_head(features).reshape(-1),
        }

    def _apply_token_dropout(self, x: torch.Tensor) -> torch.Tensor:
        drop_prob = float(self.cfg.model.token_dropout)
        if not self.training or drop_prob <= 0 or x.shape[1] <= 1:
            return x
        keep = torch.rand(x.shape[:2], device=x.device) > drop_prob
        force_keep = torch.randint(0, x.shape[1], (x.shape[0],), device=x.device)
        keep[torch.arange(x.shape[0], device=x.device), force_keep] = True
        return x * keep.unsqueeze(-1).to(dtype=x.dtype) / max(1.0 - drop_prob, 1e-6)

    def _task_attention_heads(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = {}
        attentions = {}
        for task, head in [("incidence", self.incidence_head), ("progression", self.progression_head)]:
            weights = torch.softmax(self.task_pool_score[task](x).squeeze(-1), dim=1)
            pooled = (x * weights.unsqueeze(-1)).sum(dim=1)
            outputs[task] = head(self.norm(pooled)).reshape(-1)
            attentions[task] = {
                token: float(weights[:, index].mean().detach().cpu())
                for index, token in enumerate(self.tokens)
            }
        self._last_attention_weights = attentions
        return outputs

    def _anchor_index(self) -> int:
        if "m30_pa" in self.tokens:
            return self.tokens.index("m30_pa")
        return 0

    def _anchor_attention_heads(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        anchor_index = self._anchor_index()
        anchor = x[:, anchor_index]
        context_indices = [index for index in range(x.shape[1]) if index != anchor_index]
        outputs = {}
        attentions = {}

        for task, head in [("incidence", self.incidence_head), ("progression", self.progression_head)]:
            if not context_indices:
                fused = anchor
                attentions[task] = {self.tokens[anchor_index]: 1.0}
            else:
                context_x = x[:, context_indices]
                weights = torch.softmax(self.task_pool_score[task](context_x).squeeze(-1), dim=1)
                context = (context_x * weights.unsqueeze(-1)).sum(dim=1)
                scale = torch.sigmoid(self.context_scale_logits[task])
                fused = anchor + scale * context

                total = 1.0 + float(scale.detach().cpu())
                task_attention = {self.tokens[anchor_index]: 1.0 / total}
                for local_index, token_index in enumerate(context_indices):
                    task_attention[self.tokens[token_index]] = (
                        float(scale.detach().cpu()) / total * float(weights[:, local_index].mean().detach().cpu())
                    )
                attentions[task] = task_attention
            outputs[task] = head(self.norm(fused)).reshape(-1)

        self._last_attention_weights = attentions
        return outputs

    def forward(self, images: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.architecture == "patch_mil":
            return self._forward_patch_mil(images)

        embeddings = self._build_base_embeddings(images)
        first_key = self.required_images[0]
        batch_size = images[first_key].shape[0]
        device = images[first_key].device

        token_tensors = []
        for token in self.tokens:
            gated_embedding = embeddings[token] * self._token_gate(token)
            token_tensors.append(gated_embedding + self._metadata_embedding(token, batch_size, device))
        x = torch.stack(token_tensors, dim=1)
        x = self._apply_token_dropout(x)

        if self.cfg.model.fusion_type == "anchor_attention":
            x = self.transformer(x)
            return self._anchor_attention_heads(x)

        if self.cfg.model.fusion_type == "task_attention":
            x = self.transformer(x)
            return self._task_attention_heads(x)

        if self.cfg.model.fusion_type == "gated_pooling":
            return self._task_attention_heads(x)

        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.transformer(x)
        cls_out = self.norm(x[:, 0])
        return self._heads(cls_out)

    def _extract_patch_tokens(self, image: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = image.shape
        expected = self.patch_grid_size * self.patch_size
        if height != expected or width != expected:
            raise ValueError(
                f"patch_mil expected {expected}x{expected} images, got {height}x{width}."
            )
        patches = image.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(
            batch_size * self.num_patches,
            channels,
            self.patch_size,
            self.patch_size,
        )
        patch_embeddings = self.image_encoder(patches)
        return patch_embeddings.reshape(batch_size, self.num_patches, -1)

    def _forward_patch_mil(self, images: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        image = images[self.required_images[0]]
        tokens = self._extract_patch_tokens(image)
        valid_mask = self._patch_valid_mask(tokens.device)
        tokens = tokens + self.patch_pos_embedding
        tokens = tokens * valid_mask.view(1, -1, 1).to(dtype=tokens.dtype)
        tokens = self._apply_token_dropout(tokens)
        padding_mask = ~valid_mask.view(1, -1).expand(tokens.shape[0], -1)
        tokens = self.patch_transformer(tokens, src_key_padding_mask=padding_mask)
        tokens = tokens * valid_mask.view(1, -1, 1).to(dtype=tokens.dtype)

        outputs = {}
        attentions = {}
        patch_names = [
            f"r{row}_c{col}"
            for row in range(self.patch_grid_size)
            for col in range(self.patch_grid_size)
        ]
        for task, head in [("incidence", self.incidence_head), ("progression", self.progression_head)]:
            scores = self.task_pool_score[task](tokens).squeeze(-1)
            scores = scores.masked_fill(~valid_mask.view(1, -1), torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=1)
            pooled = (tokens * weights.unsqueeze(-1)).sum(dim=1)
            outputs[task] = head(self.norm(pooled)).reshape(-1)
            attentions[task] = {
                patch_names[index]: float(weights[:, index].mean().detach().cpu())
                for index in range(self.num_patches)
            }
        self._last_attention_weights = attentions
        return outputs

    def _patch_valid_mask(self, device: torch.device) -> torch.Tensor:
        grid = self.patch_grid_size
        mask = torch.ones((grid, grid), dtype=torch.bool, device=device)
        image_key = self.required_images[0]
        if image_key.endswith("_pa"):
            mask[:2, :] = False
            mask[-2:, :] = False
        elif image_key.endswith("_lat"):
            mask[:2, :4] = False
            mask[6, :] = False
            mask[4:6, 5:7] = False
        return mask.reshape(-1)
