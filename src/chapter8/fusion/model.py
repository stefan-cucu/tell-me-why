import json
import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .classifier import load_classifier


FUSION_STATE_FILE = "fusion_modules.pt"
FUSION_CONFIG_FILE = "fusion_config.json"
LLM_ADAPTER_DIR = "llm_adapter"


@dataclass
class FusionConfig:
    embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    classifier_path: str = ""
    llm_hidden_size: int = 4096
    encoder_hidden_size: int = 1024
    num_attention_heads: int = 32
    cross_attn_hidden_size: int = 512
    num_cross_attention_heads: int = 8
    num_memory_tokens: int = 64
    cross_attn_layers: tuple = (16, 20, 24, 28, 32, 35)
    aux_ce_weight: float = 0.2
    normalize_classifier_embedding: bool = True
    classifier_threshold: float = 0.5

    def to_dict(self):
        data = self.__dict__.copy()
        data["cross_attn_layers"] = list(data["cross_attn_layers"])
        return data

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        if "cross_attn_layers" in data:
            data["cross_attn_layers"] = tuple(data["cross_attn_layers"])
        return cls(**data)


class MemoryResampler(nn.Module):
    def __init__(self, encoder_hidden_size, llm_hidden_size, num_heads, num_memory_tokens):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_memory_tokens, encoder_hidden_size) / math.sqrt(encoder_hidden_size))
        self.query_norm = nn.LayerNorm(encoder_hidden_size)
        self.kv_norm = nn.LayerNorm(encoder_hidden_size)
        self.cross_attn = nn.MultiheadAttention(encoder_hidden_size, num_heads, batch_first=True)
        self.post_norm = nn.LayerNorm(encoder_hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(encoder_hidden_size, encoder_hidden_size * 4),
            nn.GELU(),
            nn.Linear(encoder_hidden_size * 4, encoder_hidden_size),
        )
        self.output_proj = nn.Linear(encoder_hidden_size, llm_hidden_size)

    def forward(self, encoder_hidden_states, encoder_attention_mask=None):
        kv = encoder_hidden_states
        batch_size = kv.shape[0]
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)
        key_padding_mask = None
        if encoder_attention_mask is not None:
            key_padding_mask = encoder_attention_mask == 0

        norm_kv = self.kv_norm(kv)
        attn_out, _ = self.cross_attn(
            self.query_norm(latents),
            norm_kv,
            norm_kv,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        latents = latents + attn_out
        latents = latents + self.mlp(self.post_norm(latents))
        return self.output_proj(latents)


class FusionLayer(nn.Module):
    

    def __init__(self, original_layer, hidden_size, num_heads, layer_idx=None, cross_attn_hidden_size=512):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = original_layer.self_attn
        self.mlp = original_layer.mlp
        self.input_layernorm = original_layer.input_layernorm
        self.post_attention_layernorm = original_layer.post_attention_layernorm
        self.attention_type = original_layer.attention_type
        self.classifier_cross_norm = nn.LayerNorm(hidden_size)
        self.classifier_memory_norm = nn.LayerNorm(hidden_size)
        self.classifier_cross_q_proj = nn.Linear(hidden_size, cross_attn_hidden_size)
        self.classifier_cross_k_proj = nn.Linear(hidden_size, cross_attn_hidden_size)
        self.classifier_cross_v_proj = nn.Linear(hidden_size, cross_attn_hidden_size)
        self.classifier_cross_out_proj = nn.Linear(cross_attn_hidden_size, hidden_size)
        self.classifier_cross_attn = nn.MultiheadAttention(cross_attn_hidden_size, num_heads, batch_first=True)
        self.classifier_cross_gate = nn.Parameter(torch.tensor(-4.0))


    def _causal_lm(self):
        
        if hasattr(self.llm, "base_model") and hasattr(self.llm.base_model, "model"):
            return self.llm.base_model.model
        return self.llm

    def _compact_sft_forward(self, input_ids, attention_mask, labels, classifier_memory, **kwargs):
        causal_lm = self._causal_lm()
        decoder_outputs = causal_lm.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            classifier_memory=classifier_memory,
            return_dict=True,
            **kwargs,
        )
        hidden_states = decoder_outputs.last_hidden_state

        if labels is None:
            logits = causal_lm.lm_head(hidden_states[:, -1:, :])
            return None, logits, 0

        supervised_positions = (labels != -100).nonzero(as_tuple=False)
        if supervised_positions.numel() == 0:
            logits = causal_lm.lm_head(hidden_states[:, -1:, :])
            return logits.sum() * 0.0, logits, 0

        valid = supervised_positions[:, 1] > 0
        supervised_positions = supervised_positions[valid]
        if supervised_positions.numel() == 0:
            logits = causal_lm.lm_head(hidden_states[:, -1:, :])
            return logits.sum() * 0.0, logits, 0

        batch_idx = supervised_positions[:, 0]
        label_pos = supervised_positions[:, 1]
        pred_hidden = hidden_states[batch_idx, label_pos - 1]
        target = labels[batch_idx, label_pos].to(pred_hidden.device)

        chunk_size = 128
        loss_sum = None
        token_count = 0
        last_logits = None
        for start in range(0, pred_hidden.shape[0], chunk_size):
            end = min(start + chunk_size, pred_hidden.shape[0])
            chunk_logits = causal_lm.lm_head(pred_hidden[start:end])
            chunk_target = target[start:end]
            chunk_loss = F.cross_entropy(chunk_logits.float(), chunk_target, ignore_index=-100, reduction="sum")
            loss_sum = chunk_loss if loss_sum is None else loss_sum + chunk_loss
            token_count += int((chunk_target != -100).sum().item())
            last_logits = chunk_logits.detach()

        if token_count == 0:
            loss = pred_hidden.sum() * 0.0
        else:
            loss = loss_sum / token_count
        return loss, last_logits, token_count

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        classifier_memory = kwargs.pop("classifier_memory", None)
        kwargs.pop("classifier_memory_mask", None)

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        if classifier_memory is not None:
            residual = hidden_states
            query_states = self.classifier_cross_norm(hidden_states)
            memory = self.classifier_memory_norm(classifier_memory.to(dtype=query_states.dtype, device=query_states.device))
            query = self.classifier_cross_q_proj(query_states)
            key = self.classifier_cross_k_proj(memory)
            value = self.classifier_cross_v_proj(memory)
            cross_out, _ = self.classifier_cross_attn(query, key, value, need_weights=False)
            cross_out = self.classifier_cross_out_proj(cross_out)
            gate = torch.sigmoid(self.classifier_cross_gate).to(dtype=cross_out.dtype)
            hidden_states = residual + gate * cross_out

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class FusionModel(nn.Module):
    def __init__(self, llm, embedding_model, classifier, config: FusionConfig):
        super().__init__()
        self.llm = llm
        self.embedding_model = embedding_model
        self.classifier = classifier
        self.fusion_config = config

        self.memory_resampler = MemoryResampler(
            encoder_hidden_size=config.encoder_hidden_size,
            llm_hidden_size=config.llm_hidden_size,
            num_heads=config.num_attention_heads,
            num_memory_tokens=config.num_memory_tokens,
        )
        self.mlp_h512_proj = nn.Linear(512, config.llm_hidden_size)
        self.mlp_h256_proj = nn.Linear(256, config.llm_hidden_size)
        self.mlp_logits_proj = nn.Linear(2, config.llm_hidden_size)
        self.decision_proj = nn.Sequential(
            nn.Linear(4, config.llm_hidden_size),
            nn.GELU(),
            nn.Linear(config.llm_hidden_size, config.llm_hidden_size * 2),
        )
        nn.init.zeros_(self.decision_proj[-1].weight)
        nn.init.zeros_(self.decision_proj[-1].bias)
        self.memory_aux_head = nn.Linear(config.llm_hidden_size, 2)

        self.last_fusion_metrics = {}
        self.last_classifier_decision = {}
        self._patch_decoder_layers()
        self._freeze_side_models()

    @property
    def config(self):
        return self.llm.config


    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.embedding_model.eval()
            self.classifier.eval()
        return self

    def _freeze_side_models(self):
        self.embedding_model.eval()
        self.classifier.eval()
        for param in self.embedding_model.parameters():
            param.requires_grad_(False)
        for param in self.classifier.parameters():
            param.requires_grad_(False)

    def _find_decoder_layers(self):
        for _, module in self.llm.named_modules():
            layers = getattr(module, "layers", None)
            if isinstance(layers, nn.ModuleList) and len(layers) > 0:
                if "DecoderLayer" in layers[0].__class__.__name__ or hasattr(layers[0], "self_attn"):
                    return layers
        raise RuntimeError("Could not locate Qwen decoder layers on the LLM.")

    def _patch_decoder_layers(self):
        layers = self._find_decoder_layers()
        max_idx = len(layers) - 1
        for idx in self.fusion_config.cross_attn_layers:
            if idx < 0 or idx > max_idx:
                raise ValueError(f"Cross-attention layer index {idx} is outside 0..{max_idx}")
            if isinstance(layers[idx], FusionLayer):
                continue
            layers[idx] = FusionLayer(
                layers[idx],
                hidden_size=self.fusion_config.llm_hidden_size,
                num_heads=self.fusion_config.num_cross_attention_heads,
                layer_idx=idx,
                cross_attn_hidden_size=self.fusion_config.cross_attn_hidden_size,
            )

    def fusion_module_names(self):
        return (
            "memory_resampler",
            "mlp_h512_proj",
            "mlp_h256_proj",
            "mlp_logits_proj",
            "decision_proj",
            "memory_aux_head",
            "classifier_cross_norm",
            "classifier_memory_norm",
            "classifier_cross_q_proj",
            "classifier_cross_k_proj",
            "classifier_cross_v_proj",
            "classifier_cross_out_proj",
            "classifier_cross_attn",
            "classifier_cross_gate",
        )

    def fusion_state_dict(self):
        keep = self.fusion_module_names()
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if any(name in k for name in keep)}

    def load_fusion_state_dict(self, state_dict, strict=True):
        own = self.state_dict()
        missing = []
        for key, value in state_dict.items():
            if key not in own:
                if strict:
                    missing.append(key)
                continue
            own[key].copy_(value.to(device=own[key].device, dtype=own[key].dtype))
        if missing:
            raise RuntimeError(f"Unknown fusion state keys: {missing[:5]}")


    def cross_attention_gate_values(self):
        values = {}
        for idx, layer in enumerate(self._find_decoder_layers()):
            if isinstance(layer, FusionLayer):
                layer_idx = layer.layer_idx if layer.layer_idx is not None else idx
                values[f"fusion/gate_layer_{layer_idx}"] = torch.sigmoid(layer.classifier_cross_gate.detach()).float().item()
        return values

    def trainable_parameters(self):
        for name, param in self.named_parameters():
            if any(marker in name for marker in self.fusion_module_names()):
                yield param
            elif "lora_" in name:
                yield param

    def move_fusion_modules(self, device, dtype=None):
        modules = [
            self.memory_resampler,
            self.mlp_h512_proj,
            self.mlp_h256_proj,
            self.mlp_logits_proj,
            self.decision_proj,
            self.memory_aux_head,
        ]
        for module in modules:
            module.to(device=device, dtype=dtype)
        self.classifier.to(device=device)
        for layer in self._find_decoder_layers():
            if isinstance(layer, FusionLayer):
                layer.classifier_cross_norm.to(device=device, dtype=dtype)
                layer.classifier_memory_norm.to(device=device, dtype=dtype)
                layer.classifier_cross_q_proj.to(device=device, dtype=dtype)
                layer.classifier_cross_k_proj.to(device=device, dtype=dtype)
                layer.classifier_cross_v_proj.to(device=device, dtype=dtype)
                layer.classifier_cross_out_proj.to(device=device, dtype=dtype)
                layer.classifier_cross_attn.to(device=device, dtype=dtype)
                layer.classifier_cross_gate.data = layer.classifier_cross_gate.data.to(device=device, dtype=dtype or layer.classifier_cross_gate.dtype)

    def _last_token_pool(self, hidden_states, attention_mask):
        lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
        batch_idx = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        return hidden_states[batch_idx, lengths]

    def build_decision_features(self, classifier_logits, decision_labels=None, decision_confidences=None):
        classifier_probs = torch.softmax(classifier_logits.float(), dim=-1)
        classifier_pred = (classifier_probs[:, 1] >= self.fusion_config.classifier_threshold).long()
        classifier_confidence = torch.where(
            classifier_pred == 1,
            classifier_probs[:, 1],
            classifier_probs[:, 0],
        )

        if decision_labels is None:
            decision_label = classifier_pred
            p_benign = classifier_probs[:, 0]
            p_malware = classifier_probs[:, 1]
            decision_confidence = classifier_confidence
        else:
            decision_label = decision_labels.to(classifier_logits.device).long()
            valid = decision_label != -100
            safe_label = decision_label.clamp(min=0, max=1)
            if decision_confidences is None:
                decision_confidence = torch.ones_like(classifier_probs[:, 0])
            else:
                decision_confidence = decision_confidences.to(classifier_logits.device).float().clamp(0.0, 1.0)
            p_benign = torch.where(safe_label == 0, decision_confidence, 1.0 - decision_confidence)
            p_malware = torch.where(safe_label == 1, decision_confidence, 1.0 - decision_confidence)
            p_benign = torch.where(valid, p_benign, classifier_probs[:, 0])
            p_malware = torch.where(valid, p_malware, classifier_probs[:, 1])
            decision_label = torch.where(valid, safe_label, classifier_pred)
            decision_confidence = torch.where(valid, decision_confidence, classifier_confidence)

        margin = (p_malware - p_benign).abs()
        decision_features = torch.stack([p_benign, p_malware, decision_confidence, margin], dim=-1)
        return {
            "features": decision_features,
            "label": decision_label,
            "confidence": decision_confidence,
            "classifier_probs": classifier_probs,
            "classifier_label": classifier_pred,
            "classifier_confidence": classifier_confidence,
        }

    def encode_classifier_memory(self, encoder_input_ids, encoder_attention_mask, decision_labels=None, decision_confidences=None):
        with torch.no_grad():
            encoder_outputs = self.embedding_model(
                input_ids=encoder_input_ids,
                attention_mask=encoder_attention_mask,
                output_hidden_states=False,
                return_dict=True,
            )
            encoder_hidden = encoder_outputs.last_hidden_state
            pooled = self._last_token_pool(encoder_hidden, encoder_attention_mask)
            if self.fusion_config.normalize_classifier_embedding:
                pooled = F.normalize(pooled.float(), p=2, dim=-1).to(dtype=encoder_hidden.dtype)
            classifier_features = self.classifier(pooled.float())

        memory = self.memory_resampler(encoder_hidden, encoder_attention_mask)
        dtype = memory.dtype
        decision = self.build_decision_features(
            classifier_features["logits"],
            decision_labels=decision_labels,
            decision_confidences=decision_confidences,
        )
        decision_tokens = self.decision_proj(decision["features"].to(memory.device, dtype=dtype)).view(
            memory.shape[0], 2, memory.shape[-1]
        )
        h512_token = self.mlp_h512_proj(classifier_features["h512"].to(memory.device, dtype=dtype)).unsqueeze(1)
        h256_token = self.mlp_h256_proj(classifier_features["h256"].to(memory.device, dtype=dtype)).unsqueeze(1)
        logits_token = self.mlp_logits_proj(classifier_features["logits"].to(memory.device, dtype=dtype)).unsqueeze(1)
        memory = torch.cat([decision_tokens, h512_token, h256_token, logits_token, memory], dim=1)
        aux_logits = self.memory_aux_head(memory.mean(dim=1))
        return memory, aux_logits, classifier_features["logits"], decision

    @torch.no_grad()
    def classifier_decision(self, encoder_input_ids, encoder_attention_mask):
        _, _, _, decision = self.encode_classifier_memory(encoder_input_ids, encoder_attention_mask)
        return decision


    def _causal_lm(self):
        
        if hasattr(self.llm, "base_model") and hasattr(self.llm.base_model, "model"):
            return self.llm.base_model.model
        return self.llm

    def _compact_sft_forward(self, input_ids, attention_mask, labels, classifier_memory, **kwargs):
        causal_lm = self._causal_lm()
        decoder_outputs = causal_lm.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            classifier_memory=classifier_memory,
            return_dict=True,
            **kwargs,
        )
        hidden_states = decoder_outputs.last_hidden_state

        if labels is None:
            logits = causal_lm.lm_head(hidden_states[:, -1:, :])
            return None, logits, 0

        supervised_positions = (labels != -100).nonzero(as_tuple=False)
        if supervised_positions.numel() == 0:
            logits = causal_lm.lm_head(hidden_states[:, -1:, :])
            return logits.sum() * 0.0, logits, 0

        valid = supervised_positions[:, 1] > 0
        supervised_positions = supervised_positions[valid]
        if supervised_positions.numel() == 0:
            logits = causal_lm.lm_head(hidden_states[:, -1:, :])
            return logits.sum() * 0.0, logits, 0

        batch_idx = supervised_positions[:, 0]
        label_pos = supervised_positions[:, 1]
        pred_hidden = hidden_states[batch_idx, label_pos - 1]
        target = labels[batch_idx, label_pos].to(pred_hidden.device)

        chunk_size = 128
        loss_sum = None
        token_count = 0
        last_logits = None
        for start in range(0, pred_hidden.shape[0], chunk_size):
            end = min(start + chunk_size, pred_hidden.shape[0])
            chunk_logits = causal_lm.lm_head(pred_hidden[start:end])
            chunk_target = target[start:end]
            chunk_loss = F.cross_entropy(chunk_logits.float(), chunk_target, ignore_index=-100, reduction="sum")
            loss_sum = chunk_loss if loss_sum is None else loss_sum + chunk_loss
            token_count += int((chunk_target != -100).sum().item())
            last_logits = chunk_logits.detach()

        if token_count == 0:
            loss = pred_hidden.sum() * 0.0
        else:
            loss = loss_sum / token_count
        return loss, last_logits, token_count

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        encoder_input_ids=None,
        encoder_attention_mask=None,
        classifier_labels=None,
        decision_labels=None,
        decision_confidences=None,
        **kwargs,
    ):
        if encoder_input_ids is None or encoder_attention_mask is None:
            raise ValueError("encoder_input_ids and encoder_attention_mask are required for fusion forward")
        memory, aux_logits, frozen_classifier_logits, decision = self.encode_classifier_memory(
            encoder_input_ids,
            encoder_attention_mask,
            decision_labels=decision_labels,
            decision_confidences=decision_confidences,
        )
        sft_loss, compact_logits, supervised_token_count = self._compact_sft_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            classifier_memory=memory,
            **kwargs,
        )
        loss = sft_loss
        aux_loss = None
        metrics = {}
        if sft_loss is not None:
            metrics["fusion/sft_loss"] = sft_loss.detach().float().item()
        metrics["fusion/supervised_tokens"] = float(supervised_token_count)
        if classifier_labels is not None:
            labels_for_metrics = classifier_labels.to(aux_logits.device)
            valid = labels_for_metrics != -100
            if valid.any():
                aux_preds = aux_logits.detach().argmax(dim=-1)
                frozen_preds = frozen_classifier_logits.detach().to(aux_logits.device).argmax(dim=-1)
                metrics["fusion/aux_accuracy"] = (aux_preds[valid] == labels_for_metrics[valid]).float().mean().item()
                metrics["fusion/frozen_classifier_accuracy"] = (frozen_preds[valid] == labels_for_metrics[valid]).float().mean().item()
            if self.fusion_config.aux_ce_weight > 0:
                aux_loss = F.cross_entropy(aux_logits.float(), labels_for_metrics, ignore_index=-100)
                metrics["fusion/aux_loss"] = aux_loss.detach().float().item()
                if loss is None:
                    loss = self.fusion_config.aux_ce_weight * aux_loss
                else:
                    loss = loss + self.fusion_config.aux_ce_weight * aux_loss
        metrics["fusion/memory_tokens"] = float(memory.shape[1])
        metrics["fusion/decision_confidence_mean"] = decision["confidence"].detach().float().mean().item()
        metrics["fusion/memory_l2_mean"] = memory.detach().float().norm(dim=-1).mean().item()
        metrics.update(self.cross_attention_gate_values())
        self.last_fusion_metrics = metrics
        return {
            "loss": loss,
            "logits": compact_logits,
            "aux_logits": aux_logits,
            "aux_loss": aux_loss,
            "frozen_classifier_logits": frozen_classifier_logits,
        }

    @torch.no_grad()
    def generate(self, encoder_input_ids=None, encoder_attention_mask=None, **generation_kwargs):
        if encoder_input_ids is None or encoder_attention_mask is None:
            raise ValueError("encoder_input_ids and encoder_attention_mask are required for fusion generation")

        input_ids = generation_kwargs.pop("input_ids")
        attention_mask = generation_kwargs.pop("attention_mask", torch.ones_like(input_ids))
        max_new_tokens = int(generation_kwargs.pop("max_new_tokens", 128))
        eos_token_id = generation_kwargs.pop("eos_token_id", None)
        pad_token_id = generation_kwargs.pop("pad_token_id", eos_token_id)
        do_sample = bool(generation_kwargs.pop("do_sample", False))
        temperature = generation_kwargs.pop("temperature", None)
        top_p = generation_kwargs.pop("top_p", None)
        repetition_penalty = float(generation_kwargs.pop("repetition_penalty", 1.0) or 1.0)
        no_repeat_ngram_size = int(generation_kwargs.pop("no_repeat_ngram_size", 0) or 0)
        stop_on_complete_json = bool(generation_kwargs.pop("stop_on_complete_json", False))
        decode_tokenizer = generation_kwargs.pop("decode_tokenizer", None)
        completion_start_idx = generation_kwargs.pop("completion_start_idx", input_ids.shape[1])
        stop_check_interval = int(generation_kwargs.pop("stop_check_interval", 8) or 8)
        stop_strings = generation_kwargs.pop("stop_strings", ())
        if isinstance(stop_strings, str):
            stop_strings = (stop_strings,)
        else:
            stop_strings = tuple(stop_strings or ())
        generation_kwargs.pop("use_cache", None)

        def has_complete_json_after_think(text):
            if "</think>" not in text:
                return False
            tail = text.split("</think>", 1)[1]
            start = tail.find("{")
            if start < 0:
                return False
            depth = 0
            in_string = False
            escape = False
            for ch in tail[start:]:
                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_string = False
                    continue
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return True
            return False

        memory, _, _, decision = self.encode_classifier_memory(encoder_input_ids, encoder_attention_mask)
        self.last_classifier_decision = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in decision.items()}
        generated = input_ids
        current_input_ids = input_ids
        past_key_values = None
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

        for _ in range(max_new_tokens):
            outputs = self.llm(
                input_ids=current_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                classifier_memory=memory,
                logits_to_keep=1,
                return_dict=True,
            )
            logits = outputs.logits[:, -1, :]
            if repetition_penalty != 1.0:
                for batch_idx in range(generated.shape[0]):
                    seen_tokens = torch.unique(generated[batch_idx])
                    token_scores = logits[batch_idx, seen_tokens]
                    token_scores = torch.where(
                        token_scores < 0,
                        token_scores * repetition_penalty,
                        token_scores / repetition_penalty,
                    )
                    logits[batch_idx, seen_tokens] = token_scores
            if no_repeat_ngram_size > 1:
                ngram_start = int(completion_start_idx)
                for batch_idx in range(generated.shape[0]):
                    seq = generated[batch_idx, ngram_start:].tolist()
                    if len(seq) < no_repeat_ngram_size - 1:
                        continue
                    suffix = tuple(seq[-(no_repeat_ngram_size - 1):])
                    banned_tokens = set()
                    limit = len(seq) - no_repeat_ngram_size + 1
                    for pos in range(max(limit, 0)):
                        ngram = tuple(seq[pos:pos + no_repeat_ngram_size])
                        if ngram[:-1] == suffix:
                            banned_tokens.add(ngram[-1])
                    if banned_tokens:
                        logits[batch_idx, list(banned_tokens)] = -float("inf")

            if do_sample:
                if temperature is not None and temperature > 0:
                    logits = logits / temperature
                probs = torch.softmax(logits.float(), dim=-1)
                if top_p is not None and 0 < top_p < 1:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                    cumulative = torch.cumsum(sorted_probs, dim=-1)
                    remove = cumulative > top_p
                    remove[..., 1:] = remove[..., :-1].clone()
                    remove[..., 0] = False
                    sorted_probs = sorted_probs.masked_fill(remove, 0.0)
                    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    next_sorted = torch.multinomial(sorted_probs, num_samples=1)
                    next_token = sorted_idx.gather(-1, next_sorted)
                else:
                    next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            if eos_token_id is not None:
                replacement = torch.full_like(next_token, pad_token_id if pad_token_id is not None else eos_token_id)
                next_token = torch.where(finished.unsqueeze(-1), replacement, next_token)

            generated = torch.cat([generated, next_token], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
            past_key_values = outputs.past_key_values
            current_input_ids = next_token

            if eos_token_id is not None:
                finished = finished | (next_token.squeeze(-1) == eos_token_id)
                if bool(finished.all()):
                    break

            if (stop_on_complete_json or stop_strings) and decode_tokenizer is not None and generated.shape[0] == 1:
                generated_tokens = generated.shape[1] - int(completion_start_idx)
                if generated_tokens > 0 and generated_tokens % max(stop_check_interval, 1) == 0:
                    completion = decode_tokenizer.decode(
                        generated[0][int(completion_start_idx):],
                        skip_special_tokens=True,
                    )
                    if stop_strings and any(stop_string in completion for stop_string in stop_strings):
                        break
                    if stop_on_complete_json and has_complete_json_after_think(completion):
                        break

        return generated

    def save_pretrained(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.llm.save_pretrained(os.path.join(output_dir, LLM_ADAPTER_DIR))
        torch.save(self.fusion_state_dict(), os.path.join(output_dir, FUSION_STATE_FILE))
        with open(os.path.join(output_dir, FUSION_CONFIG_FILE), "w") as f:
            json.dump(self.fusion_config.to_dict(), f, indent=2)


def build_fusion_model(
    llm,
    embedding_model,
    classifier_path,
    llm_hidden_size=None,
    encoder_hidden_size=None,
    num_attention_heads=None,
    cross_attn_hidden_size=512,
    num_cross_attention_heads=8,
    num_memory_tokens=64,
    cross_attn_layers=(16, 20, 24, 28, 32, 35),
    aux_ce_weight=0.2,
    embedding_model_name="Qwen/Qwen3-Embedding-0.6B",
    classifier_threshold=0.5,
):
    llm_config = llm.config
    emb_config = embedding_model.config
    config = FusionConfig(
        embedding_model_name=embedding_model_name,
        classifier_path=classifier_path,
        llm_hidden_size=llm_hidden_size or llm_config.hidden_size,
        encoder_hidden_size=encoder_hidden_size or emb_config.hidden_size,
        num_attention_heads=num_attention_heads or llm_config.num_attention_heads,
        cross_attn_hidden_size=cross_attn_hidden_size,
        num_cross_attention_heads=num_cross_attention_heads,
        num_memory_tokens=num_memory_tokens,
        cross_attn_layers=tuple(cross_attn_layers),
        aux_ce_weight=aux_ce_weight,
        classifier_threshold=classifier_threshold,
    )
    classifier = load_classifier(classifier_path)
    return FusionModel(llm, embedding_model, classifier, config)


def load_fusion_state(model, fusion_dir):
    state_path = os.path.join(fusion_dir, FUSION_STATE_FILE)
    if not os.path.exists(state_path):
        raise FileNotFoundError(state_path)
    state = torch.load(state_path, map_location="cpu")
    model.load_fusion_state_dict(state, strict=False)
    return model
