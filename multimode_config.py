from transformers import Qwen2Config


class Config(Qwen2Config):
    def __init__(
            self,
            text_emb_dim=312,
            bill_text_max_length=50,
            minipro_text_max_length=17,
            spm_text_max_length=20,
            app_text_max_length=15,
            sousuo_text_max_length=12,

            paybill_feat_dim=67,
            paybill_item_num=250,
            minipro_item_num=150,
            spm_item_num=150,
            sousuo_item_num=10,
            app_item_num=100,
            tabular_item_num=50,
            baseinfo_feat_dim=762,
            tabular_feat_dim=1711,

            event_label_num=71,
            event_feat_dim=204,
            event_label_encoding_dim=256,

            modal_num=4,
            layer_num=6,
            head_num=4,
            hidden_dim=256,
            output_dim=128,
            fusion_input_dim=1536,
            fusion_output_dim=512,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.text_emb_dim = text_emb_dim

        self.bill_text_max_length = bill_text_max_length
        self.minipro_text_max_length = minipro_text_max_length
        self.spm_text_max_length = spm_text_max_length
        self.app_text_max_length = app_text_max_length
        self.sousuo_text_max_length = sousuo_text_max_length

        self.paybill_feat_dim = paybill_feat_dim
        self.tabular_feat_dim = tabular_feat_dim

        self.paybill_item_num = paybill_item_num
        self.minipro_item_num = minipro_item_num
        self.spm_item_num = spm_item_num
        self.sousuo_item_num = sousuo_item_num
        self.app_item_num = app_item_num
        self.tabular_item_num = tabular_item_num

        self.modal_item_num_dict = {
            'bill_text_desc': paybill_item_num,
            'minipro_text_desc': minipro_item_num,
            'spm_text_desc': spm_item_num,
            'sousuo_text_desc': sousuo_item_num,
            'app_text_desc': app_item_num,
            'tabular_desc': tabular_item_num,
        }

        self.baseinfo_feat_dim = baseinfo_feat_dim
        self.event_label_num = event_label_num
        self.event_feat_dim = event_feat_dim
        self.event_label_encoding_dim = event_label_encoding_dim
        self.modal_num = modal_num
        self.layer_num = layer_num
        self.head_num = head_num
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fusion_input_dim = fusion_input_dim
        self.fusion_output_dim = fusion_output_dim

        # 模态 tokenid
        self.bill_token_id = 151665
        self.minipro_token_id = 151666
        self.spm_token_id = 151667
        self.sousuo_token_id = 151668
        self.app_token_id = 151669
        self.tabualr_token_id = 151670
        self.modal_token_id_dict = {
            'bill_text_desc': 151665,
            'minipro_text_desc': 151666,
            'spm_text_desc': 151667,
            'sousuo_text_desc': 151668,
            'app_text_desc': 151669,
            'tabular_desc': 151670,
        }


if __name__ == "__main__":
    config = Config()
    print(config)