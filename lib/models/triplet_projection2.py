from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

#加入confidence损失 进行 对新的 confidence 语义在训练阶段不被破坏

#你这个损失不是整体不收敛，而是 Beta KL loss 卡在了一个由参数化方式造成的不可达下界。

class RobertaTripletProjectionRegressor(torch.nn.Module):
    """
    Decoder-only LLM backbone with:
      1. human/edited/AI classification;
      2. scalar edit-degree prediction in [0, 1];
      3. Human-relative magnitude supervision for Edited text.

    The historical class name is retained to avoid breaking imports.
    """

    def __init__(
        self,
        model_path: str,
        proj_dim: int = 256,
        num_labels: int = 3,
        dropout: float = 0.1,
        proto_weight: float = 0.3,
        proto_momentum: float = 0.95,
        teacher_kappa_min: float = 2.0,
        teacher_kappa_max: float = 30.0,
        beta_eps: float = 1e-4,
        teacher_y_eps: float = 0.01,
        use_lora: bool = True,
        freeze_backbone: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules=None,
    ):
        super().__init__()

        if not 0.0 <= proto_weight <= 1.0:
            raise ValueError("proto_weight must be in [0, 1].")
        if not 0.0 <= proto_momentum < 1.0:
            raise ValueError("proto_momentum must be in [0, 1).")

        self.proto_weight = proto_weight
        self.proto_momentum = proto_momentum
        self.teacher_kappa_min = teacher_kappa_min
        self.teacher_kappa_max = teacher_kappa_max
        self.beta_eps = beta_eps
        self.teacher_y_eps = teacher_y_eps
        self.use_lora = use_lora
        self.freeze_backbone = freeze_backbone

        # self.backbone = AutoModelForCausalLM.from_pretrained(
        #     model_path,
        #     torch_dtype="auto",
        #     trust_remote_code=True,
        # )
        
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        self.backbone.config.use_cache = False

        config = self.backbone.config
        hidden_size = (
            getattr(config, "hidden_size", None)
            or getattr(config, "n_embd", None)
            or getattr(config, "d_model", None)
        )
        if hidden_size is None:
            raise ValueError("Cannot infer hidden size from the backbone configuration.")

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        if use_lora:
            try:
                from peft import LoraConfig, get_peft_model
            except ImportError as exc:
                raise ImportError("use_lora=True requires the peft package.") from exc

            targets = lora_target_modules or self._infer_lora_target_modules()
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=targets,
            )
            self.backbone = get_peft_model(self.backbone, lora_config)
            self.backbone.print_trainable_parameters()

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, proj_dim),
            torch.nn.Tanh(),
        )

        # Student Beta distribution over edit degree.
        self.degree_dist_head = torch.nn.Sequential(
            torch.nn.Linear(proj_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(128, 2),
        )

        self.classifier = torch.nn.Linear(proj_dim, num_labels)
        self.prototype_classifier = torch.nn.Sequential(
            torch.nn.Linear(proj_dim + 4, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(128, num_labels),
        )

        self.register_buffer("human_proto", torch.zeros(proj_dim))
        self.register_buffer("ai_proto", torch.zeros(proj_dim))
        self.register_buffer("proto_initialized", torch.tensor(False))

    def _infer_lora_target_modules(self):
        model_type = getattr(self.backbone.config, "model_type", "").lower()
        if "gptj" in model_type or "gpt-j" in model_type:
            return ["q_proj", "k_proj", "v_proj", "out_proj"]
        return ["q_proj", "k_proj", "v_proj", "o_proj"]

    def encode(self, input_ids, attention_mask):
        """Use the last valid token hidden state as the text representation."""
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]
        last_index = attention_mask.long().sum(dim=1).sub(1).clamp(min=0)
        batch_index = torch.arange(hidden.size(0), device=hidden.device)
        last_hidden = hidden[batch_index, last_index]
        return F.normalize(self.proj(last_hidden.float()), p=2, dim=-1)

    @torch.no_grad()
    def update_segment_prototypes(self, z_h, z_a):
        batch_h = F.normalize(z_h.detach().mean(dim=0), dim=0)
        batch_a = F.normalize(z_a.detach().mean(dim=0), dim=0)

        if not bool(self.proto_initialized.item()):
            self.human_proto.copy_(batch_h)
            self.ai_proto.copy_(batch_a)
            self.proto_initialized.fill_(True)
            return

        momentum = self.proto_momentum
        self.human_proto.mul_(momentum).add_((1.0 - momentum) * batch_h)
        self.ai_proto.mul_(momentum).add_((1.0 - momentum) * batch_a)
        self.human_proto.copy_(F.normalize(self.human_proto, dim=0))
        self.ai_proto.copy_(F.normalize(self.ai_proto, dim=0))

    def get_segment_features(self, z, eps: float = 1e-12):
        z = F.normalize(z, dim=-1)
        p_h = F.normalize(self.human_proto, dim=0).to(z.device)
        p_a = F.normalize(self.ai_proto, dim=0).to(z.device)
        axis = p_a - p_h
        denom = torch.sum(axis * axis).clamp(min=eps)

        alpha_raw = torch.sum((z - p_h) * axis, dim=-1) / denom
        alpha = alpha_raw.clamp(0.0, 1.0)
        projected = p_h + alpha.unsqueeze(-1) * axis
        residual = torch.norm(z - projected, dim=-1)
        d_h = torch.norm(z - p_h, dim=-1)
        d_a = torch.norm(z - p_a, dim=-1)

        geometry = torch.stack((alpha, residual, d_h, d_a), dim=-1)
        return torch.cat((z, geometry), dim=-1), {
            "alpha": alpha,
            "alpha_raw": alpha_raw,
            "residual": residual,
            "d_h": d_h,
            "d_a": d_a,
        }

    def classify(self, z, return_aux: bool = False):
        base_logits = self.classifier(z)
        if not bool(self.proto_initialized.item()):
            return (base_logits, None, None) if return_aux else base_logits

        prototype_input, aux = self.get_segment_features(z)
        prototype_logits = self.prototype_classifier(prototype_input)
        logits = (1.0 - self.proto_weight) * base_logits + self.proto_weight * prototype_logits
        return (logits, prototype_logits, aux) if return_aux else logits

    def predict_beta_params2(self, z):
        values = F.softplus(self.degree_dist_head(z)) + 1.0 + self.beta_eps
        return values[:, 0], values[:, 1]
    
    
    def predict_beta_params(self, z):
        raw = self.degree_dist_head(z).float()
        mu_raw, kappa_raw = raw.unbind(dim=-1)

        # 与 Teacher 的 target 范围保持一致：[0.01, 0.99]
        mu = self.teacher_y_eps + (
            1.0 - 2.0 * self.teacher_y_eps
        ) * torch.sigmoid(mu_raw)

        # 与 Teacher 的 concentration 范围保持一致：[2, 30]
        kappa = self.teacher_kappa_min + (
            self.teacher_kappa_max - self.teacher_kappa_min
        ) * torch.sigmoid(kappa_raw)

        alpha = (mu * kappa).clamp_min(self.beta_eps)
        beta = ((1.0 - mu) * kappa).clamp_min(self.beta_eps)

        return alpha, beta


    def beta_mean(self, alpha, beta):
        return alpha / (alpha + beta + self.beta_eps)

    def predict_degree_from_z(self, z):
        alpha, beta = self.predict_beta_params(z)
        return self.beta_mean(alpha, beta)

    def degree_head(self, z):
        return self.predict_degree_from_z(z).unsqueeze(-1)

    def make_teacher_beta(self, target, confidence):
        target = target.clamp(self.teacher_y_eps, 1.0 - self.teacher_y_eps)
        confidence = confidence.clamp(0.0, 1.0)
        kappa = self.teacher_kappa_min + confidence * (
            self.teacher_kappa_max - self.teacher_kappa_min
        )
        alpha = (target * kappa).clamp(min=self.beta_eps)
        beta = ((1.0 - target) * kappa).clamp(min=self.beta_eps)
        return alpha, beta

    @staticmethod
    def beta_kl_divergence(alpha_t, beta_t, alpha_s, beta_s, eps: float = 1e-8):
        alpha_t = alpha_t.clamp(min=eps)
        beta_t = beta_t.clamp(min=eps)
        alpha_s = alpha_s.clamp(min=eps)
        beta_s = beta_s.clamp(min=eps)

        log_b_t = torch.lgamma(alpha_t) + torch.lgamma(beta_t) - torch.lgamma(alpha_t + beta_t)
        log_b_s = torch.lgamma(alpha_s) + torch.lgamma(beta_s) - torch.lgamma(alpha_s + beta_s)
        digamma_at = torch.digamma(alpha_t)
        digamma_bt = torch.digamma(beta_t)
        digamma_sum_t = torch.digamma(alpha_t + beta_t)
        return (
            log_b_s
            - log_b_t
            + (alpha_t - alpha_s) * (digamma_at - digamma_sum_t)
            + (beta_t - beta_s) * (digamma_bt - digamma_sum_t)
        )

    @staticmethod
    def human_relative_magnitude_raw(
        z_x,
        z_h,
    ):
        """Initial model-space Human-relative magnitude: ||z_x-z_h||/2."""
        return torch.norm(z_x - z_h, dim=-1) / 2.0

    @staticmethod
    def ai_corrected_magnitude_raw(
        z_x,
        z_h,
        z_a,
        eps: float = 1e-12,
    ):
        """
        Restore the original model-space AI-anchor correction:

            l_e = ||z_x - z_h|| / 2
            l_a = ||z_a - z_h|| / 2
            degree = l_e / l_a
        """
        edited_initial = torch.norm(z_x - z_h, dim=-1) / 2.0
        ai_initial = torch.norm(z_a - z_h, dim=-1) / 2.0
        
        
        return edited_initial / ai_initial.clamp(min=eps)

    @staticmethod
    def ai_alignment(
        z_x,
        z_h,
        z_a,
        eps: float = 1e-12,
    ):
        return F.cosine_similarity(
            z_x - z_h,
            z_a - z_h,
            dim=-1,
            eps=eps,
        )

    # Compatibility alias for old external calls.
    project_alpha_raw = ai_corrected_magnitude_raw

    @staticmethod
    def weighted_mean(values, weights, eps: float = 1e-12):
        return torch.sum(values * weights) / (torch.sum(weights) + eps)

    def predict_single(self, input_ids, attention_mask):
        z = self.encode(input_ids, attention_mask)
        return self.predict_degree_from_z(z), self.classify(z)

    def forward(
        self,
        h_input_ids=None,
        h_attention_mask=None,
        e_input_ids=None,
        e_attention_mask=None,
        a_input_ids=None,
        a_attention_mask=None,
        edited_degree=None,
        degree_confidence=None,
        lambda_cls: float = 1.0,
        lambda_degree: float = 1.0,
        lambda_axis: float = 1.0,
        loss_mode: str = "joint",
    ):
        z_h = self.encode(h_input_ids, h_attention_mask)
        z_e = self.encode(e_input_ids, e_attention_mask)
        z_a = self.encode(a_input_ids, a_attention_mask)

        if self.training:
            self.update_segment_prototypes(z_h, z_a)

        alpha_h_s, beta_h_s = self.predict_beta_params(z_h)
        alpha_e_s, beta_e_s = self.predict_beta_params(z_e)
        alpha_a_s, beta_a_s = self.predict_beta_params(z_a)
        degree_h = self.beta_mean(alpha_h_s, beta_h_s)
        degree_e = self.beta_mean(alpha_e_s, beta_e_s)
        degree_a = self.beta_mean(alpha_a_s, beta_a_s)

        logits_h, _, aux_h = self.classify(z_h, return_aux=True)
        logits_e, _, aux_e = self.classify(z_e, return_aux=True)
        logits_a, _, aux_a = self.classify(z_a, return_aux=True)

        output = {
            "z_h": z_h,
            "z_e": z_e,
            "z_a": z_a,
            "degree_h": degree_h,
            "degree_e": degree_e,
            "degree_a": degree_a,
            "logits_h": logits_h,
            "logits_e": logits_e,
            "logits_a": logits_a,
            "proto_alpha_h": None if aux_h is None else aux_h["alpha"],
            "proto_alpha_e": None if aux_e is None else aux_e["alpha"],
            "proto_alpha_a": None if aux_a is None else aux_a["alpha"],
        }

        if edited_degree is None:
            output["loss"] = None
            return output
        
        batch_size = z_h.size(0)
        device = z_h.device

        # =========================================================
        # 1. 三类编辑度目标
        # =========================================================
        target_h = torch.zeros(
            batch_size,
            device=device,
        )

        target_e = edited_degree.to(
            device=device,
            dtype=target_h.dtype,
        ).clamp(
            0.0,
            1.0,
        )

        target_a = torch.ones(
            batch_size,
            device=device,
        )

        # =========================================================
        # 2. 原始 confidence
        #
        # 用于控制 Teacher Beta 的分布集中程度。
        # =========================================================
        confidence_e = (
            torch.ones(
                batch_size,
                device=device,
                dtype=target_e.dtype,
            )
            if degree_confidence is None
            else degree_confidence.to(
                device=device,
                dtype=target_e.dtype,
            ).clamp(
                0.0,
                1.0,
            )
        )

        confidence_h = torch.ones_like(
            target_h
        )

        confidence_a = torch.ones_like(
            target_a
        )

        # =========================================================
        # 3. Loss 权重
        #
        # 原始 confidence 已经通过 kappa 影响 Teacher Beta。
        # 使用 sqrt 映射减轻二次加权导致的过度削弱。
        # =========================================================
        loss_weight_e = (
            0.05
            + 0.95
            * torch.sqrt(
                confidence_e
            )
        )

        # =========================================================
        # 4. 构造 Teacher Beta
        # =========================================================
        teacher_h = self.make_teacher_beta(
            target_h,
            confidence_h,
        )

        teacher_e = self.make_teacher_beta(
            target_e,
            confidence_e,
        )

        teacher_a = self.make_teacher_beta(
            target_a,
            confidence_a,
        )

        # =========================================================
        # 5. Beta KL loss
        # =========================================================
        kl_h = self.beta_kl_divergence(
            *teacher_h,
            alpha_h_s,
            beta_h_s,
        )

        kl_e = self.beta_kl_divergence(
            *teacher_e,
            alpha_e_s,
            beta_e_s,
        )

        kl_a = self.beta_kl_divergence(
            *teacher_a,
            alpha_a_s,
            beta_a_s,
        )

        kl_e_loss = (
            kl_e * loss_weight_e
        ).mean()
        
        
        beta_kl_loss = (
            kl_h.mean()
            + kl_e_loss
            + kl_a.mean()
        ) / 3.0


        # =========================================================
        # 6. Degree regression loss
        # =========================================================
        degree_h_loss = F.smooth_l1_loss(
            degree_h,
            target_h,
            beta=0.1,
        )

        degree_e_each = F.smooth_l1_loss(
            degree_e,
            target_e,
            beta=0.1,
            reduction="none",
        )

        degree_e_loss = (
            degree_e_each * loss_weight_e
        ).mean()


        degree_a_loss = F.smooth_l1_loss(
            degree_a,
            target_a,
            beta=0.1,
        )

        degree_loss = (
            degree_h_loss
            + degree_e_loss
            + degree_a_loss
        ) / 3.0

        # =========================================================
        # 7. 模型空间 AI-anchor loss
        # =========================================================
        anchor_degree_raw = (
            self.ai_corrected_magnitude_raw(
                z_e,
                z_h,
                z_a,
            )
        )

        anchor_each = F.smooth_l1_loss(
            anchor_degree_raw,
            target_e,
            beta=0.1,
            reduction="none",
        )

        # anchor_degree_loss = self.weighted_mean(
        #     anchor_each,
        #     loss_weight_e,
        # )
        anchor_degree_loss = (
            anchor_each * loss_weight_e
        ).mean()
        
        alignment_e = self.ai_alignment(
            z_e,
            z_h,
            z_a,
        )
        labels_h = torch.zeros(batch_size, dtype=torch.long, device=device)
        labels_e = torch.ones(batch_size, dtype=torch.long, device=device)
        labels_a = torch.full((batch_size,), 2, dtype=torch.long, device=device)
        
        classification_loss = (
            F.cross_entropy(logits_h, labels_h)
            + F.cross_entropy(logits_e, labels_e)
            + F.cross_entropy(logits_a, labels_a)
        ) / 3.0

        mapping_loss = beta_kl_loss + degree_loss
        
        if loss_mode == "cls":
            loss = lambda_cls * classification_loss
        elif loss_mode == "degree":
            loss = lambda_degree * mapping_loss + lambda_axis * anchor_degree_loss
        elif loss_mode == "joint":
            loss = (
                lambda_cls * classification_loss
                + lambda_degree * mapping_loss
                + lambda_axis * anchor_degree_loss
            )
        else:
            raise ValueError(f"Unsupported loss_mode: {loss_mode}")

        output.update({
            "loss": loss,
            "classification_loss": (
                classification_loss.detach()
            ),
            "mapping_loss": (
                mapping_loss.detach()
            ),
            "beta_kl_loss": (
                beta_kl_loss.detach()
            ),
            "degree_loss": (
                degree_loss.detach()
            ),
            "anchor_degree_loss": (
                anchor_degree_loss.detach()
            ),
            "anchor_degree_raw_mean": (
                anchor_degree_raw.detach().mean()
            ),
            "ai_alignment_mean": (
                alignment_e.detach().mean()
            ),

            # 新增
            "degree_confidence_mean": (
                confidence_e.detach().mean()
            ),
            "degree_confidence_min": (
                confidence_e.detach().min()
            ),
            "degree_confidence_max": (
                confidence_e.detach().max()
            ),
            "degree_loss_weight_mean": (
                loss_weight_e.detach().mean()
            ),

            "axis_degree_loss": (
                anchor_degree_loss.detach()
            ),
            "axis_degree_raw_mean": (
                anchor_degree_raw.detach().mean()
            ),
        })
        return output
