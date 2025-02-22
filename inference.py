import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import json
import argparse
import numpy as np
import soundfile as sf
import librosa
import torch
import pyloudnorm as pyln
import tqdm
import traceback

from models import load_model_with_args
from functions import load_ola_func_with_args
from utils import loudnorm, str2bool, db2linear

import warnings
warnings.filterwarnings("ignore")

model = None
continuous_nnet = None
device = torch.device("cpu")


def main():
    global model, continuous_nnet, device

    parser = argparse.ArgumentParser(description="Inference for MedleyVox", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=60))
    parser.add_argument("--target", type=str, default="vocals")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--model_dir", type=str, default="checkpoints", help="model directory")
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--use_overlapadd", type=str, default=None, choices=[None, "ola", "ola_norm", "w2v", "w2v_chunk", "sf_chunk"], help="use overlapadd functions, ola, ola_norm, w2v will work with ola_window_len, ola_hop_len argugments. w2v_chunk and sf_chunk is chunk-wise processing based on VAD, so you have to specify the vad_method args. If you use sf_chunk (spectral_featrues_chunk), you also need to specify spectral_features.")
    parser.add_argument("--vad_method", type=str, default="spec", choices=["spec", "webrtc"], help="what method do you want to use for 'voice activity detection (vad) -- split chunks -- processing. Only valid when 'w2v_chunk' or 'sf_chunk' for args.use_overlapadd.")
    parser.add_argument("--spectral_features", type=str, default="mfcc", choices=["mfcc", "spectral_centroid"], help="what spectral feature do you want to use in correlation calc in speaker assignment (only valid when using sf_chunk)")
    parser.add_argument("--w2v_ckpt_dir", type=str, default="pretrained", help="only valid when use_overlapadd is 'w2v' or 'w2v_chunk'.")
    parser.add_argument("--w2v_nth_layer_output", nargs="+", type=int, default=[0], help="wav2vec nth layer output")
    parser.add_argument("--ola_window_len", type=float, default=None, help="ola window size in [sec]")
    parser.add_argument("--ola_hop_len", type=float, default=None, help="ola hop size in [sec]")
    parser.add_argument("--use_ema_model", type=str2bool, default=True, help="use ema model or online model? only vaind when args.ema it True (model trained with ema)")
    parser.add_argument("--mix_consistent_out", type=str2bool, default=True, help="only valid when the model is trained with mixture_consistency loss. Default is True.")
    parser.add_argument("--reorder_chunks", type=str2bool, default=True, help="ola reorder chunks")
    parser.add_argument("--inference_data_dir", type=str, default="inputs", help="data where you want to separate")
    parser.add_argument("--results_save_dir", type=str, default="results", help="save directory")
    parser.add_argument("--output_format", type=str, default="wav", choices=["wav", "mp3", "flac"], help="output format")
    parser.add_argument("--separate_storage", type=str2bool, default=False, help="save results in separate folders with the same name as the input file")
    parser.add_argument("--skip_error", type=str2bool, default=True, help="skip error files while separating instead of stopping")
    args, _ = parser.parse_known_args()

    args.exp_result_dir = os.path.join(args.model_dir, args.exp_name)
    with open(os.path.join(args.exp_result_dir, f"{args.target}.json"), "r") as f:
        args_dict = json.load(f)
    for key, value in args_dict["args"].items():
        setattr(args, key, value)

    if args.use_overlapadd == "w2v" or args.use_overlapadd == "w2v_chunk":
        assert os.path.exists(os.path.join(args.w2v_ckpt_dir, "xlsr_53_56k.pt")), "Please download the pretrained model xlsr_53_56k.pt from https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr_53_56k.pt and put it in the 'pretrained' folder!"

    # load model architecture
    model = load_model_with_args(args)

    if torch.cuda.is_available() and args.use_gpu:
        device = torch.device("cuda")
        print("Using GPU for inference")
    else:
        print("Using CPU for inference")

    target_model_path = f"{args.exp_result_dir}/{args.target}.pth"
    checkpoint = torch.load(target_model_path, map_location=device)
    if args.ema and args.use_ema_model:
        print("Use ema model")
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        checkpoint = {
            k.replace("ema_model.module.", ""): v
            for k, v in checkpoint.items()
            if k.replace("ema_model.module.", "") in model_dict
        }
        # 2. overwrite entries in the existing state dict
        model_dict.update(checkpoint)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
    elif args.ema and not args.use_ema_model:
        print("Use ema online model")
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        checkpoint = {
            k.replace("online_model.module.", ""): v
            for k, v in checkpoint.items()
            if k.replace("online_model.module.", "") in model_dict
        }
        # 2. overwrite entries in the existing state dict
        model_dict.update(checkpoint)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    meter = pyln.Meter(args.sample_rate)

    if args.use_overlapadd:
        continuous_nnet = load_ola_func_with_args(args, model, device, meter)

    data_list = os.listdir(args.inference_data_dir)
    print(f"Total files found: {len(data_list)}")
    data_list = tqdm.tqdm(data_list, desc="Separating")

    success_files = []
    error_files = []
    for data_path in data_list:
        song_name = os.path.basename(data_path).split(".")[0]
        data_list.set_postfix_str(song_name)

        if args.separate_storage:
            save_dir = os.path.join(args.results_save_dir, song_name)
        else:
            save_dir = args.results_save_dir
        os.makedirs(save_dir, exist_ok=True)

        try:
            mixture, _ = librosa.load(
                path=os.path.join(args.inference_data_dir, data_path),
                sr=args.sample_rate,
                mono=False,
                dtype=np.float32
            )
        except Exception as e:
            print(f"Error loading {data_path}, error: {e}, skipping...")
            traceback.print_exc()
            error_files.append(data_path)
            continue

        try:
            if len(mixture.shape) != 1 and mixture.shape[0] > 2:
                mixture = np.mean(mixture, axis=0)

            if len(mixture.shape) == 2:
                left = mixture[0, :]
                right = mixture[1, :]
                left_out_wav_1, left_out_wav_2 = inference(args, left, meter)
                right_out_wav_1, right_out_wav_2 = inference(args, right, meter)
                out_wav_1 = np.stack([left_out_wav_1, right_out_wav_1], axis=-1)
                out_wav_2 = np.stack([left_out_wav_2, right_out_wav_2], axis=-1)
            else:
                out_wav_1, out_wav_2 = inference(args, mixture, meter)

            save_wav_path_1 = os.path.join(save_dir, f"{song_name}_output_1.{args.output_format}")
            save_wav_path_2 = os.path.join(save_dir, f"{song_name}_output_2.{args.output_format}")

            sf.write(save_wav_path_1, out_wav_1, args.sample_rate)
            sf.write(save_wav_path_2, out_wav_2, args.sample_rate)
            success_files.append(data_path)
        except Exception as e:
            if args.skip_error:
                print(f"Error separating {data_path}, error: {e}, skipping...")
                traceback.print_exc()
                error_files.append(data_path)
                continue
            else:
                raise e

    print("Separation done!")
    if len(success_files) == 0:
        print("No files separated")
    else:
        print(f"Successfully separated {len(success_files)} files: {success_files}")
    if len(error_files) == 0:
        print("No files with errors")
    else:
        print(f"Error in separating {len(error_files)} files: {error_files}")

def inference(args, mix, meter):
    assert (model is not None or continuous_nnet is not None), "model is not loaded"

    mix, adjusted_gain = loudnorm(mix, -24.0, meter)
    mix = np.expand_dims(mix, axis=0)
    mix = mix.reshape(1, mix.shape[0], mix.shape[1])
    mix = torch.as_tensor(mix, dtype=torch.float32).to(device)

    if args.use_overlapadd:
        out_wavs = continuous_nnet.forward(mix)
    else:
        out_wavs = model.separate(mix)

    if args.use_gpu:
        out_wav_1 = out_wavs[0, 0, :].cpu().detach().numpy()
        out_wav_2 = out_wavs[0, 1, :].cpu().detach().numpy()
    else:
        out_wav_1 = out_wavs[0, 0, :]
        out_wav_2 = out_wavs[0, 1, :]

    out_wav_1 = out_wav_1 * db2linear(-adjusted_gain)
    out_wav_2 = out_wav_2 * db2linear(-adjusted_gain)

    return out_wav_1, out_wav_2


if __name__ == "__main__":
    main()
