import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss, LayerNorm
import numpy as np
import torch.distributed.nn as dist_nn
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    is_flash_attn_2_available,
    logging,
)
from transformers import Qwen2PreTrainedModel, AutoModelForCausalLM, AutoModel
from torch.utils.checkpoint import checkpoint

logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = "Query_as_Anchor_Config"


class SimpleMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc_seq = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, input_tensor):
        return self.fc_seq(input_tensor)


class LLMAdapter(nn.Module):
    def __init__(self, encoder_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.hidden_size = encoder_dim
        self.ln_q = LayerNorm(encoder_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, llm_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x))
        return x


class ZTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_heads):
        super(ZTransformer, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.transformer_layer = nn.TransformerEncoderLayer(hidden_dim, num_heads)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)
        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, src, src_mask=None):
        src = self.encoder(src)
        src = src.permute(1, 0, 2)

        output = self.transformer(src, src_key_padding_mask=src_mask)
        output = output.permute(1, 0, 2)
        output = self.decoder(output)
        return output


class ModalEncoder(nn.Module):
    def __init__(self, config):
        super(ModalEncoder, self).__init__()
        self.modal_encoder_name = config.modal_encoder_name
        self.sentence_model = AutoModel.from_pretrained(config.modal_encoder_name)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids, attention_mask=None):
        sentence_embeddings = self.sentence_model(input_ids, attention_mask=attention_mask)
        sentence_embeddings = sentence_embeddings.last_hidden_state[:, 0]
        return sentence_embeddings


class BillEncoder(nn.Module):
    def __init__(self, config):
        super(BillEncoder, self).__init__()
        self.config = config
        self.text_model = ModalEncoder(config)
        self.llm_adapter = LLMAdapter(self.config.text_emb_dim, self.config.hidden_size)

    def forward(self, batch):
        bill_text_item_tensor, bill_text_item_padnum_tensor, bill_text_item_mask_tensor = batch
        valid_uid_index = bill_text_item_padnum_tensor > 0
        valid_text_index = (~bill_text_item_mask_tensor).reshape(-1)

        if valid_uid_index.sum() == 0:
            total_cls_emb = torch.zeros(valid_uid_index.size(0), self.config.paybill_item_num,
                                        self.config.text_emb_dim).to(valid_uid_index.device)
            # adapter
            output_emb = self.llm_adapter(total_cls_emb)
            return output_emb, total_cls_emb.mean(1)
            # Bert Model
        input_ids, attention_masks = bill_text_item_tensor
        input_ids = input_ids.reshape(-1, self.config.bill_text_max_length)[valid_text_index]
        attention_masks = attention_masks.reshape(-1, self.config.bill_text_max_length)[valid_text_index]
        cls_emb = self.text_model(input_ids=input_ids, attention_mask=attention_masks)

        total_cls_emb = torch.zeros(valid_text_index.size(0), self.config.text_emb_dim, dtype=cls_emb.dtype).to(
            valid_text_index.device)
        total_cls_emb[valid_text_index] = cls_emb

        total_cls_emb = total_cls_emb.reshape(-1, self.config.paybill_item_num, self.config.text_emb_dim)
        # adapter
        output_emb = self.llm_adapter(total_cls_emb)
        return output_emb, mean_pooling(total_cls_emb, ~bill_text_item_mask_tensor)


class MiniProgramEncoder(nn.Module):
    def __init__(self, config):
        super(MiniProgramEncoder, self).__init__()
        self.config = config
        self.text_model = ModalEncoder(config)
        self.llm_adapter = LLMAdapter(self.config.text_emb_dim, self.config.hidden_size)

    def forward(self, batch):
        minipro_text_item_tensor, minipro_text_item_padnum_tensor, minipro_text_item_mask_tensor = batch

        valid_uid_index = minipro_text_item_padnum_tensor > 0
        valid_text_index = (~minipro_text_item_mask_tensor).reshape(-1)

        if valid_uid_index.sum() == 0:
            total_cls_emb = torch.zeros(valid_uid_index.size(0), self.config.minipro_item_num,
                                        self.config.text_emb_dim).to(valid_uid_index.device)
            # adapter
            output_emb = self.llm_adapter(total_cls_emb)
            return output_emb, total_cls_emb.mean(1)

            # Bert Model
        input_ids, attention_masks = minipro_text_item_tensor
        input_ids = input_ids.reshape(-1, self.config.minipro_text_max_length)[valid_text_index]
        attention_masks = attention_masks.reshape(-1, self.config.minipro_text_max_length)[valid_text_index]
        cls_emb = self.text_model(input_ids=input_ids, attention_mask=attention_masks)

        total_cls_emb = torch.zeros(valid_text_index.size(0), self.config.text_emb_dim, dtype=cls_emb.dtype).to(
            valid_text_index.device)
        total_cls_emb[valid_text_index] = cls_emb

        total_cls_emb = total_cls_emb.reshape(-1, self.config.minipro_item_num, self.config.text_emb_dim)
        # adapter
        output_emb = self.llm_adapter(total_cls_emb)
        return output_emb, mean_pooling(total_cls_emb, ~minipro_text_item_mask_tensor)


class SPMEncoder(nn.Module):
    def __init__(self, config):
        super(SPMEncoder, self).__init__()
        self.config = config
        self.text_model = ModalEncoder(config)
        self.llm_adapter = LLMAdapter(self.config.text_emb_dim, self.config.hidden_size)

    def forward(self, batch):
        spm_text_item_tensor, spm_text_item_padnum_tensor, spm_text_item_mask_tensor = batch

        ## valid index extraction
        valid_uid_index = spm_text_item_padnum_tensor > 0
        valid_text_index = (~spm_text_item_mask_tensor).reshape(-1)

        if valid_uid_index.sum() == 0:
            total_cls_emb = torch.zeros(valid_uid_index.size(0), self.config.spm_item_num, self.config.text_emb_dim).to(
                valid_uid_index.device)
            # adapter
            output_emb = self.llm_adapter(total_cls_emb)
            return output_emb, total_cls_emb.mean(1)
            # Bert Model
        input_ids, attention_masks = spm_text_item_tensor
        input_ids = input_ids.reshape(-1, self.config.spm_text_max_length)[valid_text_index]
        attention_masks = attention_masks.reshape(-1, self.config.spm_text_max_length)[valid_text_index]
        cls_emb = self.text_model(input_ids=input_ids, attention_mask=attention_masks)
        total_cls_emb = torch.zeros(valid_text_index.size(0), self.config.text_emb_dim, dtype=cls_emb.dtype).to(
            valid_text_index.device)
        total_cls_emb[valid_text_index] = cls_emb

        total_cls_emb = total_cls_emb.reshape(-1, self.config.spm_item_num, self.config.text_emb_dim)
        # adapter
        output_emb = self.llm_adapter(total_cls_emb)

        return output_emb, mean_pooling(total_cls_emb, ~spm_text_item_mask_tensor)


class APPEncoder(nn.Module):
    def __init__(self, config):
        super(APPEncoder, self).__init__()
        self.config = config
        self.text_model = ModalEncoder(config)
        self.llm_adapter = LLMAdapter(self.config.text_emb_dim, self.config.hidden_size)

    def forward(self, batch):
        app_text_item_tensor, app_text_item_padnum_tensor, app_text_item_mask_tensor = batch

        valid_uid_index = app_text_item_padnum_tensor > 0
        valid_text_index = (~app_text_item_mask_tensor).reshape(-1)

        if valid_uid_index.sum() == 0:
            total_cls_emb = torch.zeros(valid_uid_index.size(0), self.config.app_item_num, self.config.text_emb_dim).to(
                valid_uid_index.device)
            # adapter
            output_emb = self.llm_adapter(total_cls_emb)
            return output_emb, total_cls_emb.mean(1)
            # Bert Model
        input_ids, attention_masks = app_text_item_tensor
        input_ids = input_ids.reshape(-1, self.config.app_text_max_length)[valid_text_index]
        attention_masks = attention_masks.reshape(-1, self.config.app_text_max_length)[valid_text_index]
        cls_emb = self.text_model(input_ids=input_ids, attention_mask=attention_masks)

        total_cls_emb = torch.zeros(valid_text_index.size(0), self.config.text_emb_dim, dtype=cls_emb.dtype).to(
            valid_text_index.device)
        total_cls_emb[valid_text_index] = cls_emb

        total_cls_emb = total_cls_emb.reshape(-1, self.config.app_item_num, self.config.text_emb_dim)

        # adapter
        output_emb = self.llm_adapter(total_cls_emb)
        return output_emb, mean_pooling(total_cls_emb, ~app_text_item_mask_tensor)


class SearchEncoder(nn.Module):
    def __init__(self, config):
        super(SearchEncoder, self).__init__()
        self.config = config
        self.text_model = ModalEncoder(config)
        self.llm_adapter = LLMAdapter(self.config.text_emb_dim, self.config.hidden_size)

    def forward(self, batch):
        sousuo_text_item_tensor, sousuo_text_item_padnum_tensor, sousuo_text_item_mask_tensor = batch

        valid_uid_index = sousuo_text_item_padnum_tensor > 0
        valid_text_index = (~sousuo_text_item_mask_tensor).reshape(-1)

        if valid_uid_index.sum() == 0:
            total_cls_emb = torch.zeros(valid_uid_index.size(0), self.config.sousuo_item_num,
                                        self.config.text_emb_dim).to(valid_uid_index.device)
            # adapter
            output_emb = self.llm_adapter(total_cls_emb)
            return output_emb, total_cls_emb.mean(1)
            # Bert Model
        input_ids, attention_masks = sousuo_text_item_tensor
        input_ids = input_ids.reshape(-1, self.config.sousuo_text_max_length)[valid_text_index]
        attention_masks = attention_masks.reshape(-1, self.config.sousuo_text_max_length)[valid_text_index]
        cls_emb = self.text_model(input_ids=input_ids, attention_mask=attention_masks)

        total_cls_emb = torch.zeros(valid_text_index.size(0), self.config.text_emb_dim, dtype=cls_emb.dtype).to(
            valid_text_index.device)
        total_cls_emb[valid_text_index] = cls_emb

        total_cls_emb = total_cls_emb.reshape(-1, self.config.sousuo_item_num, self.config.text_emb_dim)

        # adapter
        output_emb = self.llm_adapter(total_cls_emb)
        return output_emb, mean_pooling(total_cls_emb, ~sousuo_text_item_mask_tensor)


class TabularMLP(nn.Module):
    def __init__(self, config):
        super(TabularMLP, self).__init__()
        self.config = config

        self.linear = torch.nn.Linear(self.config.tabular_feat_dim, self.config.hidden_size)
        self.linear_for_prepose_pooling = torch.nn.Linear(self.config.tabular_feat_dim, self.config.text_emb_dim)

    def forward(self, batch):
        output_emb = self.linear(batch).unsqueeze(1)
        total_cls_emb = self.linear_for_prepose_pooling(batch)
        return output_emb.expand(output_emb.shape[0], self.config.tabular_item_num, output_emb.shape[-1]), total_cls_emb


def last_pooling(last_hidden_states, attention_mask=None):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def mean_pooling(token_embeddings, attention_mask=None):
    if attention_mask is None:
        return token_embeddings.mean(1)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def special_pooling(token_embeddings, input_ids):
    # 注意: 此处的 sec_user 是一个特定的token ID，使用不同tokenizer时可能需要调整
    sec_user = 151683
    tgt_idxs = (input_ids == sec_user).nonzero(as_tuple=True)[1]
    tgt_embds = token_embeddings[torch.arange(token_embeddings.size(0)), tgt_idxs]
    return tgt_embds


def cosine_similarity(x, y):
    x_norm = F.normalize(x, p=2, dim=1)
    y_norm = F.normalize(y, p=2, dim=1)
    x_sim = torch.matmul(x_norm, y_norm.t())
    return x_sim


def contrastive_loss_with_masked_negatives(X, Y, tau=0.07, margin=0.1):
    """
    实现带掩码因子的对比学习损失函数
    X: [B, d] 查询向量 q_i
    Y: [B, d] 正样本向量 d_i^+
    tau: 温度参数
    margin: 用于判断假负样本的阈值
    """
    B, d = X.shape

    # 正样本对相似度 [B]
    sim_pos = F.cosine_similarity(X, Y, dim=-1)

    # 转换为softmax分数 [B]
    exp_pos = torch.exp(sim_pos / tau)

    # 计算所有样本两两相似度 [B, B]
    all_sim_qq = cosine_similarity(X, X)
    all_sim_dd = cosine_similarity(Y, Y)
    all_sim_qd = cosine_similarity(X, Y)
    all_sim_dq = cosine_similarity(Y, X)

    # 创建掩码矩阵，排除正样本对并过滤假负样本
    mask = torch.eye(B, dtype=torch.bool, device=X.device)
    neg_mask = ~mask  # 原始负样本掩码

    # 扩展 sim_pos 以匹配 all_sim_qq 和 all_sim_dd 的形状
    sim_pos_expanded = sim_pos.unsqueeze(1).expand(-1, B)  # [B, B]

    # 计算掩码因子 m_ij
    m_ij_qq = torch.ones_like(all_sim_qq, dtype=torch.float32)
    m_ij_qq[neg_mask] = (all_sim_qq[neg_mask] <= sim_pos_expanded[neg_mask] + margin).float()

    m_ij_dd = torch.ones_like(all_sim_dd, dtype=torch.float32)
    m_ij_dd[neg_mask] = (all_sim_dd[neg_mask] <= sim_pos_expanded[neg_mask] + margin).float()

    m_ij_qd = torch.ones_like(all_sim_qd, dtype=torch.float32)
    m_ij_qd[neg_mask] = (all_sim_qd[neg_mask] <= sim_pos_expanded[neg_mask] + margin).float()

    m_ij_dq = torch.ones_like(all_sim_dq, dtype=torch.float32)
    m_ij_dq[neg_mask] = (all_sim_dq[neg_mask] <= sim_pos_expanded[neg_mask] + margin).float()

    # 提取负样本相似度，并应用掩码因子
    all_sim_qq_neg = all_sim_qq[neg_mask].view(B, -1)  # [B, B-1]
    m_ij_qq_neg = m_ij_qq[neg_mask].view(B, -1)  # [B, B-1]

    all_sim_dd_neg = all_sim_dd[neg_mask].view(B, -1)  # [B, B-1]
    m_ij_dd_neg = m_ij_dd[neg_mask].view(B, -1)  # [B, B-1]

    all_sim_qd_neg = all_sim_qd[neg_mask].view(B, -1)  # [B, B-1]
    m_ij_qd_neg = m_ij_qd[neg_mask].view(B, -1)  # [B, B-1]

    all_sim_dq_neg = all_sim_dq[neg_mask].view(B, -1)  # [B, B-1]
    m_ij_dq_neg = m_ij_dq[neg_mask].view(B, -1)  # [B, B-1]

    # 负样本softmax分数 [B, B-1]
    exp_qq_neg = torch.exp(all_sim_qq_neg / tau) * m_ij_qq_neg
    exp_dd_neg = torch.exp(all_sim_dd_neg / tau) * m_ij_dd_neg
    exp_qd_neg = torch.exp(all_sim_qd_neg / tau) * m_ij_qd_neg
    exp_dq_neg = torch.exp(all_sim_dq_neg / tau) * m_ij_dq_neg

    # 计算归一化因子 Z_i
    Z = exp_pos + exp_qq_neg.sum(dim=1) + exp_qd_neg.sum(dim=1) + exp_dq_neg.sum(dim=1)  # + exp_dd_neg.sum(dim=1)

    # 计算损失函数
    loss = -torch.log(exp_pos / Z).mean()

    return loss


class UserAdapter(nn.Module):
    def __init__(self, config):
        super(UserAdapter, self).__init__()
        self.config = config
        self.llm_adapter = LLMAdapter(self.config.text_emb_dim, self.config.hidden_size)

    def forward(self, user_emb):
        output_emb = self.llm_adapter(user_emb)
        return output_emb


class ModalAdapter(nn.Module):
    def __init__(self, config):
        super(ModalAdapter, self).__init__()
        self.config = config
        self.llm_adapter = LLMAdapter(self.config.text_emb_dim, self.config.hidden_size)

    def forward(self, modal_emb):
        output_emb = self.llm_adapter(modal_emb)
        return output_emb


class MultimodalForConditionalGeneration(Qwen2PreTrainedModel, GenerationMixin):
    def __init__(self, config, device, logit_scale_init_value=0.07):
        super().__init__(config)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            config.llm_model_name,
            torch_dtype="auto",
            device_map=device

        )
        self.modal_encoder_list = nn.ModuleDict({
            'bill': BillEncoder(config),
            'minipro': MiniProgramEncoder(config),
            'spm': SPMEncoder(config),
            'app': APPEncoder(config),
            'sousuo': SearchEncoder(config),
            'tabular': TabularMLP(config),
            'user_adapter': UserAdapter(config),
            'modal_adapter': ModalAdapter(config),
        })
        self.config = config
        self.logit_scale = config.logit_scale
        self.low_dim_mlp_left = SimpleMLP(self.config.hidden_size, 512, 128)

    def modal_emb_insert(self, input_ids, inputs_embeds, multimode_data, modal_type):
        modal_enc_name_map = {'bill_text_desc': 'bill', 'minipro_text_desc': 'minipro', 'spm_text_desc': 'spm',
                              'sousuo_text_desc': 'sousuo', 'app_text_desc': 'app', 'tabular_desc': 'tabular'}
        modal_enc_name = modal_enc_name_map[modal_type]
        modal_enc = self.modal_encoder_list[modal_enc_name]

        if self.config.gradient_checkpointing:
            modal_embeds, avg_modal_emb = checkpoint(modal_enc, multimode_data[modal_type], use_reentrant=False)
        else:
            modal_embeds, avg_modal_emb = modal_enc(multimode_data[modal_type])

        n_modal_tokens = (input_ids == self.config.modal_token_id_dict[modal_type]).sum().item()
        n_modal_features = modal_embeds.shape[0] * self.config.modal_item_num_dict[modal_type]
        if n_modal_tokens != n_modal_features:
            raise ValueError(
                f"in {modal_type}, modal features and modal tokens do not match: tokens: {n_modal_tokens}, features {n_modal_features}={modal_embeds.shape[0]}*{self.config.modal_item_num_dict[modal_type]}")

        mask = input_ids == self.config.modal_token_id_dict[modal_type]
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        modal_mask = mask_expanded.to(inputs_embeds.device)
        modal_embeds = modal_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(modal_mask, modal_embeds)

        return inputs_embeds, avg_modal_emb

    def forward(
            self,
            left_input,
            right_input,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = True,
            cache_position: Optional[torch.LongTensor] = None,
            multimode_data: Optional[Dict] = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        input_ids = left_input[0]
        attention_mask = left_input[1]

        answer_input_ids = right_input[0]
        answer_attention_mask = right_input[1]

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.llm_model.get_input_embeddings()(input_ids)
            answer_inputs_embeds = self.llm_model.get_input_embeddings()(answer_input_ids)

        if multimode_data is not None:
            modal_type_list = list(multimode_data.keys())
            avg_modal_emb_list = []
            # 替换inputs_embeds中modal token对应位置的embed
            for modal_type in modal_type_list:
                inputs_embeds, avg_modal_emb = self.modal_emb_insert(input_ids, inputs_embeds, multimode_data,
                                                                     modal_type)
                avg_modal_emb_list.append(avg_modal_emb)
            # 添加precode
            # 对每个模态的池化表征进一步池化
            avg_modal_emb_array = torch.stack(avg_modal_emb_list, dim=0)
            user_emb_prepose = avg_modal_emb_array.mean(0)
            # 前置token经过各自的adapter
            modal_emb_prepose = self.modal_encoder_list['modal_adapter'](avg_modal_emb_array)
            user_emb_prepose = self.modal_encoder_list['user_adapter'](user_emb_prepose)

            # 1+6个新的token，1为user_token，6为modal_token,剩下的是event_token
            prepos_emb_array = torch.cat([user_emb_prepose.unsqueeze(0), modal_emb_prepose], dim=0).permute(1, 0, 2)

            # 将precode插入到inputs_embeds中，并更新inputid和attmask
            # 注意: 此处ID可能随tokenizer变化
            bill_start_id = 151671
            bill_start_index = (input_ids == bill_start_id).nonzero()[0][-1].item()
            batch_size = inputs_embeds.shape[0]

            # inputs_embeds矩阵拼接
            inputs_embeds = torch.cat(
                [inputs_embeds[:, :bill_start_index, :], prepos_emb_array, inputs_embeds[:, bill_start_index:, :]],
                dim=1)
            # attentionmask 矩阵拼接
            prepose_attention_mask = torch.ones(batch_size, prepos_emb_array.shape[1], dtype=torch.long,
                                                device=attention_mask.device)
            attention_mask = torch.cat(
                [attention_mask[:, :bill_start_index], prepose_attention_mask, attention_mask[:, bill_start_index:]],
                dim=1)
            # input_ids 矩阵拼接
            prepose_input_ids = torch.ones(batch_size, prepos_emb_array.shape[1], dtype=torch.long,
                                           device=attention_mask.device)
            input_ids = torch.cat([input_ids[:, :bill_start_index], prepose_input_ids, input_ids[:, bill_start_index:]],
                                  dim=1)

        outputs_emb = self.llm_model.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position, ).last_hidden_state

        # 取base模型的表征，不带分类头
        answer_emb = self.llm_model.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=answer_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=answer_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position, ).last_hidden_state

        # hidden_states = outputs[0]
        logits = self.llm_model.lm_head(outputs_emb)

        # 定义标签
        labels = []
        assistane_b1_idx_list = []
        for input_id in input_ids:
            sep_token = 77091  # 特殊分隔符ID
            sep_index = (input_id == sep_token).nonzero()[0].item() + 2  # 跳过"助手："本身,+2 input_id[sep_index-1]=198
            assistane_b1_idx_list.append((input_id == sep_token).nonzero()[0].item() - 1)
            # 构造labels：问题部分设为-100，答案部分保留原ID
            label = input_id.clone()
            label[:sep_index] = -100
            # 过滤padding的预测loss
            label[label == 151643] = -100
            labels.append(label.unsqueeze(0))
        labels = torch.cat(labels, dim=0)
        # 右塔表征
        right_emb = last_pooling(answer_emb, answer_attention_mask)
        # 左塔表征
        left_emb = special_pooling(outputs_emb, input_ids)

        # 维度变换
        left_emb_low = self.low_dim_mlp_left(left_emb)
        right_emb_low = self.low_dim_mlp_left(right_emb)

        all_right_embs = dist_nn.all_gather(right_emb_low)  #
        all_right_embs = torch.cat(all_right_embs, dim=0)

        all_left_embs = dist_nn.all_gather(left_emb_low)
        all_left_embs = torch.cat(all_left_embs, dim=0)

        loss_contra = contrastive_loss_with_masked_negatives(all_left_embs, all_right_embs, tau=self.config.logit_scale)

        loss = None
        if labels is not None:
            logits = logits.float()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return loss, loss_contra

    def evaluate(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = True,
            cache_position: Optional[torch.LongTensor] = None,
            multimode_data: Optional[Dict] = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.llm_model.get_input_embeddings()(input_ids)

        if multimode_data is not None:
            modal_type_list = list(multimode_data.keys())
            avg_modal_emb_list = []
            for modal_type in modal_type_list:
                inputs_embeds, avg_modal_emb = self.modal_emb_insert(input_ids, inputs_embeds, multimode_data,
                                                                     modal_type)
                avg_modal_emb_list.append(avg_modal_emb)
            # 添加precode
            # 对每个模态的池化表征进一步池化
            avg_modal_emb_array = torch.stack(avg_modal_emb_list, dim=0)
            user_emb_prepose = avg_modal_emb_array.mean(0)
            # 前置token经过各自的adapter
            modal_emb_prepose = self.modal_encoder_list['modal_adapter'](avg_modal_emb_array)
            user_emb_prepose = self.modal_encoder_list['user_adapter'](user_emb_prepose)

            # 1+6个新的token，1为user_token，6为modal_token,剩下的是event_token
            prepos_emb_array = torch.cat([user_emb_prepose.unsqueeze(0), modal_emb_prepose], dim=0).permute(1, 0, 2)

            # 将precode插入到inputs_embeds中，并更新inputid和attmask
            bill_start_id = 151671
            bill_start_index = (input_ids == bill_start_id).nonzero()[0][-1].item()
            batch_size = inputs_embeds.shape[0]

            # inputs_embeds矩阵拼接
            inputs_embeds = torch.cat(
                [inputs_embeds[:, :bill_start_index, :], prepos_emb_array, inputs_embeds[:, bill_start_index:, :]],
                dim=1)
            # attentionmask 矩阵拼接
            prepose_attention_mask = torch.ones(batch_size, prepos_emb_array.shape[1], dtype=torch.long,
                                                device=attention_mask.device)
            attention_mask = torch.cat(
                [attention_mask[:, :bill_start_index], prepose_attention_mask, attention_mask[:, bill_start_index:]],
                dim=1)
            # input_ids 矩阵拼接
            prepose_input_ids = torch.ones(batch_size, prepos_emb_array.shape[1], dtype=torch.long,
                                           device=attention_mask.device)
            input_ids = torch.cat([input_ids[:, :bill_start_index], prepose_input_ids, input_ids[:, bill_start_index:]],
                                  dim=1)

        outputs_emb = self.llm_model.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position, )[0]

        left_emb = special_pooling(outputs_emb, input_ids)

        # 维度变换
        left_emb_low = self.low_dim_mlp_left(left_emb)
        left_emb_low_norm = F.normalize(left_emb_low, p=2, dim=1)
        return left_emb_low, left_emb, left_emb_low_norm