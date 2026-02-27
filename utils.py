from loguru import logger
import os
import torch


def load_model(config, model, device, used_modal_type=None):
    modal_enc_name_map = {'bill_text_desc': 'bill', 'minipro_text_desc': 'minipro', 'spm_text_desc': 'spm',
                          'sousuo_text_desc': 'sousuo', 'app_text_desc': 'app'}

    model.modal_encoder_list.load_state_dict(
        torch.load(config.lora_model_path, map_location=device)['modal_encoder_list_state_dict'])
    model.low_dim_mlp_left.load_state_dict(
        torch.load(config.lora_model_path, map_location=device)['model_state_low_dim_left'])
    lora_state_dict = torch.load(config.lora_model_path, map_location=device)['lora_state_dict']
    model.llm_model.load_state_dict(lora_state_dict, strict=False)

    return model


def save_model(config, model, optimizer, epoch, step, used_modal_type):
    llm_lora_state_dict = {k: v for k, v in model.module.llm_model.state_dict().items() if 'lora' in k}
    model_low_dim_left_dict = model.module.low_dim_mlp_left.state_dict()
    modal_encoder_list_state_dict = model.module.modal_encoder_list.state_dict()

    checkpoint = {"lora_state_dict": llm_lora_state_dict,
                  "model_state_low_dim_left": model_low_dim_left_dict,
                  "modal_encoder_list_state_dict": modal_encoder_list_state_dict,
                  "optimizer_state_dict": optimizer.state_dict(),
                  "epoch": epoch}

    save_path = config.save_path + "{}/".format(config.sub_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 移除了stage_1等硬编码
    model_save_path = save_path + "model_{}_{}_{}.pth".format(epoch, step, used_modal_type)
    torch.save(checkpoint, model_save_path)

    logger.info("save model to :{}".format(model_save_path))


# 将输入多模态数据转换到指定设备上
def multimode_data_to_device(batch, device):
    multimode_data = {}
    for k, d in batch.items():
        if k in ['llm_input_token', 'answer_label', 'user_id', 'dt', 'emb_pos_id', 'answer', 'query']:
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


if __name__ == "__main__":
    pass