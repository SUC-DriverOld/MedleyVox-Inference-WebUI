<div align="center">

# MedleyVox Inference WebUI

WebUI and notebook for [MedleyVox](https://github.com/jeonchangbin49/MedleyVox) inference.<br>
[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SUC-DriverOld/MedleyVox-Inference-WebUI/blob/master/MedleyVox.ipynb)
[![GitHub release](https://img.shields.io/github/v/release/SUC-DriverOld/MedleyVox-Inference-WebUI?label=Version)](https://github.com/SUC-DriverOld/MedleyVox-Inference-WebUI/releases/latest)
[![GitHub license](https://img.shields.io/github/license/SUC-DriverOld/MedleyVox-Inference-WebUI?label=License)](https://github.com/SUC-DriverOld/MSST-WebUI/blob/master/MedleyVox-Inference-WebUI)

</div>

## Introduction

Medley Vox is a [dataset for testing algorithms for separating multiple singers](https://arxiv.org/pdf/2211.07302) within a single music track. Also, the [authors of Medley Vox](https://github.com/jeonchangbin49/MedleyVox) proposed a neural network architecture for separating singers. However, unfortunately, they did not publish the weights. Later, their training process was [repeated by Cyru5](https://huggingface.co/Cyru5/MedleyVox/tree/main), who trained several models and published the weights in the public domain. Now this WebUI is created to use the trained models and weights for inference.

## Usage

[Click here]((https://colab.research.google.com/github/SUC-DriverOld/MedleyVox-Inference-WebUI/blob/master/MedleyVox.ipynb)) to run the webUI on Google Colab. You can also run this code on your local machine by installing the requirements and running the `webui.py` file. For Windows users, I will provide a one-click pakage in the future.

- Clone this repository.

    ```bash
    git clone https://github.com/SUC-DriverOld/MedleyVox-Inference-WebUI
    cd MedleyVox-Inference-WebUI
    ```

- Install the requirements. The version of Python is recommended to be 3.10. Here we use conda to install the requirements.

    ```bash
    conda create -n medleyvox python=3.10 -y
    conda activate medleyvox
    # Make sure the pip version is below 24.1, for example, 24.0
    python -m pip install --upgrade pip==24.0 setuptools
    pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
    ```

- Download MedleyVox models.

    [Cyru5](https://huggingface.co/Cyru5) has trained several models and published the weights on HuggingFace. You can download the weights from [here](https://huggingface.co/Cyru5/MedleyVox/tree/main). The weights are stored in the `checkpoints` folder in folder format, with each model folder containing a model file (.pth) and its corresponding configuration file (.json). For example:
    ```bash
    checkpoints
        ├── vocals 238
        │   ├── vocals.pth
        │   └── vocals.json
        ├── multi_singing_librispeech_138
        │   ├── vocals.pth
        │   └── vocals.json
        ├── singing_librispeech_ft_iSRNet
        │   └── ...
        └── ...
    ```

- Download the pretrained model for overlapadd.

    If you use overlapadd and the choice of model is 'w2v' or 'w2v_chunk', you need to download the pretrained model [xlsr_53_56k.pt](https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr_53_56k.pt) and put it in the 'pretrained' folder. For example:
    ```bash
    pretrained
        └── xlsr_53_56k.pt
    ```

- Run the webUI. After running the webUI, it will automatically open a web page in your default browser.

    ```bash
    python webui.py --auto_clean_cache
    ```

## Command Line

- `webui.py`: The main script of the webUI.

```
usage: webui.py [-h] [-i IP_ADDRESS] [-p PORT] [-s] [--auto_clean_cache]

WebUI for MedleyVox

options:
  -h, --help                              show this help message and exit
  -i IP_ADDRESS, --ip_address IP_ADDRESS  IP address to run the server
  -p PORT, --port PORT                    Port to run the server
  -s, --share                             Enable gradio share link
  --auto_clean_cache                      Auto clean WebUI cache
```

- `inference.py`: The script for inference.

```
usage: inference.py [-h] [--target TARGET] [--exp_name EXP_NAME] [--model_dir MODEL_DIR] [--use_gpu USE_GPU] [--use_overlapadd {None,ola,ola_norm,w2v,w2v_chunk,sf_chunk}] [--vad_method {spec,webrtc}]
                    [--spectral_features {mfcc,spectral_centroid}] [--w2v_ckpt_dir W2V_CKPT_DIR] [--w2v_nth_layer_output W2V_NTH_LAYER_OUTPUT [W2V_NTH_LAYER_OUTPUT ...]] [--ola_window_len OLA_WINDOW_LEN]
                    [--ola_hop_len OLA_HOP_LEN] [--use_ema_model USE_EMA_MODEL] [--mix_consistent_out MIX_CONSISTENT_OUT] [--reorder_chunks REORDER_CHUNKS] [--inference_data_dir INFERENCE_DATA_DIR]
                    [--results_save_dir RESULTS_SAVE_DIR] [--output_format {wav,mp3,flac}] [--separate_storage SEPARATE_STORAGE] [--skip_error SKIP_ERROR]

Inference for MedleyVox

options:
  -h, --help                                                show this help message and exit
  --target TARGET
  --exp_name EXP_NAME
  --model_dir MODEL_DIR                                     model directory
  --use_gpu USE_GPU
  --use_overlapadd {None,ola,ola_norm,w2v,w2v_chunk,sf_chunk}
                                                            use overlapadd functions, ola, ola_norm, w2v will work with ola_window_len, ola_hop_len argugments. w2v_chunk and sf_chunk is chunk-wise processing based on VAD, so you have to specify the vad_method args. If you use sf_chunk (spectral_featrues_chunk), you also need to specify spectral_features.
  --vad_method {spec,webrtc}                                what method do you want to use for 'voice activity detection (vad) -- split chunks -- processing. Only valid when 'w2v_chunk' or 'sf_chunk' for args.use_overlapadd.
  --spectral_features {mfcc,spectral_centroid}              what spectral feature do you want to use in correlation calc in speaker assignment (only valid when using sf_chunk)
  --w2v_ckpt_dir W2V_CKPT_DIR                               only valid when use_overlapadd is 'w2v' or 'w2v_chunk'.
  --w2v_nth_layer_output W2V_NTH_LAYER_OUTPUT [W2V_NTH_LAYER_OUTPUT ...]
                                                            wav2vec nth layer output
  --ola_window_len OLA_WINDOW_LEN                           ola window size in [sec]
  --ola_hop_len OLA_HOP_LEN                                 ola hop size in [sec]
  --use_ema_model USE_EMA_MODEL                             use ema model or online model? only vaind when args.ema it True (model trained with ema)
  --mix_consistent_out MIX_CONSISTENT_OUT                   only valid when the model is trained with mixture_consistency loss. Default is True.
  --reorder_chunks REORDER_CHUNKS                           ola reorder chunks
  --inference_data_dir INFERENCE_DATA_DIR                   data where you want to separate
  --results_save_dir RESULTS_SAVE_DIR                       save directory
  --output_format {wav,mp3,flac}                            output format
  --separate_storage SEPARATE_STORAGE                       save results in separate folders with the same name as the input file
  --skip_error SKIP_ERROR                                   skip error files while separating instead of stopping
```

## References and Thanks

- [Medley Vox] [Repository](https://github.com/jeonchangbin49/MedleyVox) | [Paper](https://arxiv.org/pdf/2211.07302)
- [Cyru5] [Huggingface Models](https://huggingface.co/Cyru5/MedleyVox/tree/main)
- [Gradio] [Repository](https://github.com/gradio-app/gradio)