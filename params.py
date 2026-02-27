import argparse
import deepspeed


def parse_args():
    parser = argparse.ArgumentParser(description='Query as anchor')
    parser.add_argument("--num-workers", type=int, default=8, help="The number of workers for training dataloader.")
    parser.add_argument("--train-data-path", type=str, default='data/train.json', help="train data path")
    parser.add_argument("--test-data-path", type=str, default='data/test.json', help="test data path")
    parser.add_argument("--infer-window-size", type=int, default=24, help="Batch size for training per GPU.")
    parser.add_argument("--test_batch_size", type=int, default=32, help="Batch size for test per GPU.")
    parser.add_argument("--real_batch", type=int, default=2048, help="Batch size for training per GPU.")
    parser.add_argument("--context-length", type=int, default=52,
                        help="The maximum length of input text (include [CLS] & [SEP] tokens). Default to 52.")
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.0e-4, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=0.98, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=1.0e-6, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=1.0e-5, help="Weight decay.")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="Number of steps to warmup for.")
    parser.add_argument("--max_steps", type=int, default=2000000, help="Number of steps to warmup for.")
    parser.add_argument("--precision", choices=["amp", "fp16", "fp32"], default="amp", help="Floating point precision.")
    parser.add_argument('--save_path', type=str, default="./checkpoints/")
    parser.add_argument('--sub_path', type=str, default="v1_model")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--lr_scheduler_type', type=str, default="cosine")
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--used_modal_list', type=str,
                        default='bill_text_desc,app_text_desc,minipro_text_desc,sousuo_text_desc,spm_text_desc,tabular_desc')  # ,'tabular_desc'
    parser.add_argument('--token_max_length', type=int, default=1050, help="model_input max length")
    parser.add_argument('--add_new_token', type=bool, default=True,
                        help='if add new modal token')  # 只有在当前tokenizer没有modal token时开启
    parser.add_argument("--user_token_id", type=int, default=110011,
                        help='单词-对应的tokenid，该id取值和prompt使用的文本强相关，可参考tokenizer json文件')
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument('--encoder_model_path', type=str, default="./checkpoints/encoder/model.pth")
    parser.add_argument('--lora_model_path', type=str, default="./checkpoints/lora/model.pth")
    parser.add_argument('--save_steps_num', type=int, default=10000)
    parser.add_argument('--resume', type=int, default=2, help="use pretrained checkpoint")
    parser.add_argument("--alpha", type=float, default=1., help="Adam beta 1.")
    parser.add_argument('--loss_type', type=str, default="both", choices=['ntp', 'contrast', 'both'])
    parser.add_argument('--nesting_list', nargs='+', type=int, default=[128])  # [64,128,256,1024]
    parser.add_argument("--logit_scale", type=float, default=0.07, help="logit_scale")
    parser.add_argument("--answer_max_length", type=int, default=200, help="answer_max_length")
    parser.add_argument("--max_norm", type=float, default=1, help="max_norm")
    parser.add_argument("--clip_grad_flag", type=float, default=1, help="clip_grad_flag")
    parser.add_argument('--llm_model_name', type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument('--modal_encoder_name', type=str, default="thenlper/gte-base-zh")
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='Whether to enable gradient checkpointing.')
    args = parser.parse_args()
    return args


def merge_args(config, args):
    for key, value in vars(args).items():
        if value is not None:
            setattr(config, key, value)

    return config


if __name__ == '__main__':
    args = parse_args()
    print(args)