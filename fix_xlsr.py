import torch
import os
from omegaconf import DictConfig, open_dict

assert os.path.exists("pretrained/xlsr_53_56k.pt"), "Please download the pretrained model xlsr_53_56k.pt from https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr_53_56k.pt and put it in the 'pretrained' folder!"

cp_path = "pretrained/xlsr_53_56k.pt"
cp = torch.load(cp_path, map_location='cpu', weights_only=False)
wrong_key = ['eval_wer','eval_wer_config', 'eval_wer_tokenizer', 'eval_wer_post_process', 'autoregressive']
cfg = DictConfig(cp['cfg'])
with open_dict(cfg):
    for k in wrong_key:
        cfg.task.pop(k)
cp['cfg'] = cfg
torch.save(cp, "pretrained/xlsr_53_56k.pt")
print("Done")