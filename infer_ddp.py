import os

from tqdm import tqdm
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoTokenizer,
    AutoConfig,
)

from params import parse_args, merge_args
from dataset import get_data
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import load_model
from modeling import MultimodalForConditionalGeneration
from multimode_config import Config


def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def multimode_data_to_device(batch, device):
    multimode_data = {}
    for k, d in batch.items():
        if k in ['llm_input_token']:
            continue
        if isinstance(d, tuple) or isinstance(d, list):
            sub_d = []
            for dd in d:
                if isinstance(dd, tuple) or isinstance(dd, list):
                    sub_dd = [ddd.to(device) for ddd in dd]
                else:
                    sub_dd = dd.to(device)
                sub_d.append(sub_dd)
        else:
            sub_d = d.to(device)
        multimode_data[k] = sub_d
    return multimode_data


def main():
    args = parse_args()
    llm_config = AutoConfig.from_pretrained(args.llm_model_name).to_dict()
    model_encoder_config = AutoConfig.from_pretrained(args.modal_encoder_name).to_dict()
    config = Config(**llm_config)
    config = merge_args(config, args)
    config.text_emb_dim = model_encoder_config['hidden_size']
    config.used_modal_list = config.used_modal_list.split(',')

    if torch.cuda.is_available():
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        rank = int(os.environ.get('RANK', 0))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        setup(rank, world_size)

        torch.multiprocessing.set_start_method('spawn', force=True)
        config.rank = dist.get_rank()
        config.world_size = dist.get_world_size()
        config.local_device_rank = local_rank

        device = torch.device('cuda', config.local_device_rank)

    else:
        device = torch.device('cpu')

    # 文本编码器tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.modal_encoder_name, use_fast=True, cache_dir='cache')
    # llm编码器tokenizer
    llm_tokenizer = AutoTokenizer.from_pretrained(config.llm_model_name, use_fast=True, cache_dir='cache')

    if config.add_new_token:
        new_token_list = ["<|bill_pad|>", "<|minipro_pad|>", "<|spm_pad|>", "<|sousuo_pad|>", "<|app_pad|>",
                          "<|tabular_pad|>"]
        llm_tokenizer.add_tokens(new_token_list)
        new_special_token = {
            "additional_special_tokens": ["<|bill_start|>", "<|bill_end|>",
                                          '<|minipro_start|>', '<|minipro_end|>',
                                          '<|spm_start|>', '<|spm_end|>',
                                          '<|sousuo_start|>', '<|sousuo_end|>',
                                          '<|app_start|>', '<|app_end|>',
                                          '<|tabular_start|>', '<|tabular_end|>',
                                          '<|user_emb|>'
                                          ]
        }
        llm_tokenizer.add_special_tokens(new_special_token)

    test_dataloader, dataset = get_data(config, tokenizer, llm_tokenizer, config.test_data_path)
    model = MultimodalForConditionalGeneration(config, device).to(device)

    lora_config = LoraConfig(
        r=64,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  #
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    peft_model = get_peft_model(model.llm_model, lora_config)
    model.llm_model = peft_model.model

    GTE_peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,  # GTE用于特征提取
        inference_mode=False,  # 训练模式
        r=64,  # LoRA秩
        lora_alpha=32,  # LoRA alpha
        lora_dropout=0.1,  # LoRA dropout
        target_modules=["query", "value", 'key', 'output.dense']  # 可以指定特定的模块，如注意力层的query和value
    )
    model.modal_encoder_list = get_peft_model(model.modal_encoder_list, GTE_peft_config).model

    # 模型加载
    model = load_model(config, model, device)
    test_dataloader, dataset = get_data(config, tokenizer, llm_tokenizer, config.test_data_path)
    model.eval()
    for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        multimode_data = multimode_data_to_device(batch, device)
        input_ids, attention_mask = batch['llm_input_token']['input_ids'].to(device), batch['llm_input_token'][
            'attention_mask'].to(device)
        with torch.no_grad():
            left_emb_low, left_emb, left_emb_low_norm = model.evaluate(input_ids, attention_mask,
                                                                       multimode_data=multimode_data)


if __name__ == "__main__":
    main()