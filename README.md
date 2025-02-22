<div align="center">

# MedleyVox Inference WebUI

WebUI and notebook for [MedleyVox](https://github.com/jeonchangbin49/MedleyVox) inference.<br>
[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SUC-DriverOld/MedleyVox-Inference-WebUI/blob/master/MedleyVox.ipynb)
[![GitHub release](https://img.shields.io/github/v/release/SUC-DriverOld/MedleyVox-Inference-WebUI?label=Version)](https://github.com/SUC-DriverOld/MedleyVox-Inference-WebUI/releases/latest)
[![GitHub license](https://img.shields.io/github/license/SUC-DriverOld/MedleyVox-Inference-WebUI?label=License)](https://github.com/SUC-DriverOld/MedleyVox-Inference-WebUI/blob/master/LICENSE)

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
    After downloading, use the following command to fix the model (Refer to [fairseq/issues/4585](https://github.com/facebookresearch/fairseq/issues/4585)):
    ```bash
    python fix_xlsr.py
    ```

- Run the webUI. To specify the language, set environment variable `LANGUAGE="en_US"` or `LANGUAGE="zh_CN"`. To specify the IP address and port, use the `-i` and `-p` options. To enable gradio share link, use the `-s` option. To auto clean WebUI cache, use the `--auto_clean_cache. After running the webUI, it will automatically open a web page in your default browser.

    ```bash
    # Specify the language
    # Powershell:
    $env:LANGUAGE="en_US"
    # CMD:
    set LANGUAGE="en_US"
    # Bash:
    export LANGUAGE="en_US"

    # Run the webUI, for example:
    python webui.py --auto_clean_cache
    ```

> [!NOTE]
> - At present, the audio output sampling rate supported by the model is 24000kHz and cannot be changed. To solve this, you can use [AudioSR](https://github.com/haoheliu/versatile_audio_super_resolution), [Apollo](https://github.com/JusperLee/Apollo), or [Music Source Separation Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) for audio super-resolution.
> - When using WebUI on cloud platforms or Colab, please place the audio to be processed in the 'inputs' folder, and the processing results will be stored in the 'results' folder. The 'Select folder' and 'Open folder' buttons are invalid in the cloud.

### Chunk-wise processing

If the input is too long, it may be impossible to impossible due to lack of VRAM, or performance may be degraded at all. In that case, use --use_overlapadd. Among the --use_overlapadd options, "ola", "ola_norm", and "w2v" all work similarly to LambdaOverlapAdd in asteroid.

- ola: Same as LambdaOverlapAdd in asteroid.
- ola_norm: LambdaOverlapAdd with input applied chunk-wise loudness normalization (we used loudness normalization in training stage). The effect was not good. 
- w2v: When calculating the singer assignment in the overlapped region of the chunk in the LambdaOverlapAdd function based on the wave2vec2.0-xlsr model, the LambdaOverlapAdd implemented in the asteroid is simply obtained as L1 in the waveform stage. This is transformed into cosine similarity of w2v feature.
- w2v_chunk: First use VAD and divide it into chunks, then chunk-wise processing. Unlike asteroid LambdaOverlapAdd, there is no overlapped region of chunk in front and rear, so it should not be implemented as L1 distance in waveform, and the similarity in feature stage is obtained. Calculated by continuously accumulating the w2v feature for each chunk.
- sf_chunk: The principle is the same as w2v_chunk, but instead of w2v, use a spectral feature such as mfcc or spectral centroid.

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