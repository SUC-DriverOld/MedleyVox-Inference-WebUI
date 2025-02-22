__author__ = "Sucial: https://github.com/SUC-DriverOld"
__version__ = "1.0.0"

import os
import shutil
import glob
import gradio as gr
import multiprocessing
import tkinter as tk
from tkinter import filedialog
from utils import I18nAuto


os.makedirs("checkpoints", exist_ok=True)
os.makedirs("pretrained", exist_ok=True)
MODEL_DIR = "checkpoints"
PRETRAINED_MODEL_DIR = "pretrained"
TEMP_DIR = "temp"
model_list = [m for m in os.listdir(MODEL_DIR) if os.path.isdir(os.path.join(MODEL_DIR, m))]
i18n = I18nAuto(language=os.environ.get("LANGUAGE", "auto"))

def select_folder():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    selected_dir = filedialog.askdirectory()
    root.destroy()
    return selected_dir

def open_folder(folder):
    os.makedirs(folder, exist_ok=True)
    absolute_path = os.path.abspath(folder)
    if os.name == "nt":
        os.system(f"explorer {absolute_path}")
    elif os.name == "posix":
        os.system(f"xdg-open {absolute_path}")
    else:
        os.system(f"open {absolute_path}")

def change_to_audio_infer():
    return (gr.Button(i18n("Input audio files to inference"), variant="primary", visible=True),
            gr.Button(i18n("Input folder to inference"), variant="primary", visible=False))

def change_to_folder_infer():
    return (gr.Button(i18n("Input audio files to inference"), variant="primary", visible=False),
            gr.Button(i18n("Input folder to inference"), variant="primary", visible=True))

def inference_audio_fn(model_name, use_overlapadd, use_gpu, separate_storage, output_format, vad_method, spectral_features, ola_window_len, ola_hop_len, w2v_nth_layer_output, use_ema_model, mix_consistent_out, reorder_chunks, skip_error, audio_input, store_dir):
    if not audio_input:
        return i18n("Please upload at least one audio file!")
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)
    for audio in audio_input:
        shutil.copy(audio, TEMP_DIR)
    folder_input = TEMP_DIR
    message = inference_folder_fn(model_name, use_overlapadd, use_gpu, separate_storage, output_format, vad_method, spectral_features, ola_window_len, ola_hop_len, w2v_nth_layer_output, use_ema_model, mix_consistent_out, reorder_chunks, skip_error, folder_input, store_dir)
    shutil.rmtree(TEMP_DIR)
    return message

def inference_folder_fn(model_name, use_overlapadd, use_gpu, separate_storage, output_format, vad_method, spectral_features, ola_window_len, ola_hop_len, w2v_nth_layer_output, use_ema_model, mix_consistent_out, reorder_chunks, skip_error, folder_input, store_dir):
    if (use_overlapadd == "w2v" or use_overlapadd == "w2v_chunk") and not os.path.exists(os.path.join(PRETRAINED_MODEL_DIR, "xlsr_53_56k.pt")):
        return i18n("Please download the pretrained model xlsr_53_56k.pt from https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr_53_56k.pt and put it in the 'pretrained' folder!")
    if not model_name:
        return i18n("Please select a model!")
    if not store_dir:
        return i18n("Please select an output folder!")
    if not os.path.exists(folder_input):
        return i18n("Please select an input folder!")
    print("Inference started... Click 'Forced stop inference' to stop the process.")
    medleyvox_inference = multiprocessing.Process(
        target=inference,
        args=(model_name, use_overlapadd, use_gpu, separate_storage, output_format, vad_method, spectral_features, ola_window_len, ola_hop_len, w2v_nth_layer_output, use_ema_model, mix_consistent_out, reorder_chunks, skip_error, folder_input, store_dir),
        name="MedleyVox_Inference"
    )
    medleyvox_inference.start()
    print(f"Inference process started, PID: {medleyvox_inference.pid}.")
    medleyvox_inference.join()
    return i18n("Inference process finished.")

def inference(model_name, use_overlapadd, use_gpu, separate_storage, output_format, vad_method, spectral_features, ola_window_len, ola_hop_len, w2v_nth_layer_output, use_ema_model, mix_consistent_out, reorder_chunks, skip_error, folder_input, store_dir):
    print(f"Model: {model_name}, Use overlapadd: {use_overlapadd}, Use GPU: {use_gpu}, Separate storage: {separate_storage}, Output format: {output_format}, VAD method: {vad_method}, Spectral features: {spectral_features}, OLA window length: {ola_window_len}, OLA hop length: {ola_hop_len}, Wav2Vec nth layer output: {w2v_nth_layer_output}, Use EMA model: {use_ema_model}, Mix consistent output: {mix_consistent_out}, Reorder chunks: {reorder_chunks}, Skip error: {skip_error}, Folder input: {folder_input}, Store dir: {store_dir}")
    model_file = os.path.basename(glob.glob(os.path.join(MODEL_DIR, model_name, "*.pth"))[0])
    target = model_file.replace(".pth", "")
    exp_name = model_name
    model_dir = MODEL_DIR
    params = f"--target \"{target}\" --exp_name \"{exp_name}\" --model_dir \"{model_dir}\""
    if use_gpu:
        params += " --use_gpu y"
    else:
        params += " --use_gpu n"
    if use_overlapadd != "None":
        params += f" --use_overlapadd {use_overlapadd}"
    params += f" --vad_method {vad_method} --spectral_features {spectral_features} --w2v_ckpt_dir {PRETRAINED_MODEL_DIR} --w2v_nth_layer_output {w2v_nth_layer_output}"
    if ola_window_len != 0:
        params += f" --ola_window_len {ola_window_len}"
    if ola_hop_len != 0:
        params += f" --ola_hop_len {ola_hop_len}"
    if use_ema_model:
        params += " --use_ema_model y"
    else:
        params += " --use_ema_model n"
    if mix_consistent_out:
        params += " --mix_consistent_out y"
    else:
        params += " --mix_consistent_out n"
    if reorder_chunks:
        params += " --reorder_chunks y"
    else:
        params += " --reorder_chunks n"
    if skip_error:
        params += " --skip_error y"
    else:
        params += " --skip_error n"
    if separate_storage:
        params += f" --separate_storage y"
    else:
        params += f" --separate_storage n"
    params += f" --output_format {output_format} --inference_data_dir \"{folder_input}\" --results_save_dir \"{store_dir}\""
    print(params)
    os.system(f"python inference.py {params}")

def stop_inference_fn():
    for process in multiprocessing.active_children():
        if process.name == "MedleyVox_Inference":
            process.terminate()
            process.join()
            print(f"Inference process stopped, PID: {process.pid}")
            return i18n("Inference process stopped by user.")
    return i18n("No active inference process.")

def webui():
    with gr.Blocks() as demo:
        gr.Markdown(value='''<div align="center"><font size=6><b>MedleyVox-Inference-WebUI</b></font></div>''')
        gr.Markdown(value=i18n("Medley Vox is a [dataset for testing algorithms for separating multiple singers](https://arxiv.org/pdf/2211.07302) within a single music track. Also, the [authors of Medley Vox](https://github.com/jeonchangbin49/MedleyVox) proposed a neural network architecture for separating singers. However, unfortunately, they did not publish the weights. Later, their training process was [repeated by Cyru5](https://huggingface.co/Cyru5/MedleyVox/tree/main), who trained several models and published the weights in the public domain. Now this WebUI is created to use the trained models and weights for inference. Here are some precautions:<br>1. Put the [downloaded models](https://huggingface.co/Cyru5/MedleyVox) in the 'checkpoints' folder in folder format, with each model folder containing a model file (.pth) and its corresponding configuration file (.json).<br>2. If you use overlapadd and the choice of model is 'w2v' or 'w2v_chunk', you need to download the pretrained model [xlsr_53_56k.pt](https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr_53_56k.pt) and put it in the 'pretrained' folder.<br>3. At present, the audio output sampling rate supported by the model is 24000kHz and cannot be changed. To solve this, you can use [AudioSR](https://github.com/haoheliu/versatile_audio_super_resolution), [Apollo](https://github.com/JusperLee/Apollo), or [Music Source Separation Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) for audio super-resolution.<br>4. When using WebUI on cloud platforms or Colab, please place the audio to be processed in the 'inputs' folder, and the processing results will be stored in the 'results' folder. The 'Select folder' and 'Open folder' buttons are invalid in the cloud."))
        with gr.Accordion(open=True, label=i18n("Common options")):
            with gr.Row():
                with gr.Column():
                    model_name = gr.Dropdown(label=i18n("Select Model"), info=i18n("Select which model you want to use."), choices=model_list, value=None, interactive=True, scale=4)
                    use_overlapadd = gr.Radio(label=i18n("Use overlapadd"), info=i18n("Use overlapadd functions, ola, ola_norm, w2v will work with ola_window_len, ola_hop_len argugments. w2v_chunk and sf_chunk is chunk-wise processing based on VAD, so you have to specify the vad_method args. If you use sf_chunk (spectral_featrues_chunk), you also need to specify spectral_features.<br>If the input is too long, it may be impossible to inference due to lack of VRAM. In that case, use 'use_overlapadd'. Among the 'use_overlapadd' options, 'ola', 'ola_norm', and 'w2v' all work well. Use w2v_chunk or sf_chunk if these fail or as desired. You can also try experimenting with 'vad_method' options spec and webrtc when using either of the '_chunk' methods. Chunking has proven to be very useful therefore it is on by default."), choices=["None", "ola", "ola_norm", "w2v", "w2v_chunk", "sf_chunk"], value="ola", interactive=True)
                with gr.Column():
                    use_gpu = gr.Checkbox(label=i18n("Use GPU"), info=i18n("Use GPU for inference."), value=True, interactive=True)
                    separate_storage = gr.Checkbox(label=i18n("Save results in separate folders"), info=i18n("Save results in separate folders with the same name as the input file."), value=False, interactive=True)
                    skip_error = gr.Checkbox(label=i18n("Skip error files"), info=i18n("Skip error files while separating instead of stopping."), value=True, interactive=True)
                    output_format = gr.Radio(label=i18n("Output format"), info=i18n("Select the output format."), choices=["wav", "flac", "mp3"], value="wav", interactive=True)
        with gr.Accordion(open=False, label=i18n("[Click to expand] Advanced options")):
            with gr.Row():
                with gr.Column():
                    vad_method = gr.Radio(label=i18n("VAD method"), info=i18n("What method do you want to use for 'voice activity detection (vad) -- split chunks -- processing. Only valid when 'w2v_chunk' or 'sf_chunk' for args.use_overlapadd."), choices=["spec", "webrtc"], value="spec", interactive=True)
                    spectral_features = gr.Radio(label=i18n("Spectral features"), info=i18n("What spectral feature do you want to use in correlation calc in speaker assignment (only valid when using sf_chunk)"), choices=["mfcc", "spectral_centroid"], value="mfcc", interactive=True)
                    ola_window_len = gr.Number(label=i18n("OLA window length"), info=i18n("OLA window size in [sec], only valid when using ola or ola_norm. Set 0 to use the default value (None)."), value=0, interactive=True, step=0.01)
                    ola_hop_len = gr.Number(label=i18n("OLA hop length"), info=i18n("OLA hop size in [sec], only valid when using ola or ola_norm. Set 0 to use the default value (None)."), value=0, interactive=True, step=0.01)
                with gr.Column():
                    w2v_nth_layer_output = gr.Textbox(label=i18n("Wav2Vec nth layer output"), info=i18n("Wav2Vec nth layer output, only valid when using w2v or w2v_chunk. For example: 0 1 2 3, default: 0"), value="0", interactive=True)
                    use_ema_model = gr.Checkbox(label=i18n("Use EMA model"), info=i18n("Use EMA model or online model? Only vaind when args.ema it True (model trained with EMA)."), value=True, interactive=True)
                    mix_consistent_out = gr.Checkbox(label=i18n("Mix consistent output"), info=i18n("Only valid when the model is trained with mixture_consistency loss."), value=True, interactive=True)
                    reorder_chunks = gr.Checkbox(label=i18n("Reorder chunks"), info=i18n("OLA reorder chunks. Only valid when using ola or ola_norm."), value=True, interactive=True)
        with gr.Row():
            with gr.Column():
                with gr.Tabs():
                    with gr.TabItem(label=i18n("Input audio files")) as audio_tab:
                        audio_input = gr.Files(label=i18n("Input one or more audio files"), type="filepath")
                    with gr.TabItem(label=i18n("Input folder path")) as folder_tab:
                        folder_input = gr.Textbox(label=i18n("Input folder path"), info=i18n("Audio files in the folder will be used for inference."), value="inputs/", interactive=True, scale=4)
                        with gr.Row():
                            select_input_dir = gr.Button(i18n("Select folder"))
                            open_input_dir = gr.Button(i18n("Open folder"))
            with gr.Row():
                with gr.Tabs():
                    with gr.TabItem(label=i18n("Output folder path")):
                        store_dir = gr.Textbox(label=i18n("Output folder path"), info=i18n("Audio files will be saved in this folder."), value="results/", interactive=True, scale=4)
                        with gr.Row():
                            select_store_btn = gr.Button(i18n("Select folder"))
                            open_store_btn = gr.Button(i18n("Open folder"))
        inference_audio = gr.Button(i18n("Input audio files to inference"), variant="primary", visible=True)
        inference_folder = gr.Button(i18n("Input folder to inference"), variant="primary", visible=False)
        output_message = gr.Textbox(label=i18n("Output message"), interactive=False)
        stop_inference = gr.Button(i18n("Forced stop inference"), variant="stop")
        gr.Markdown('''<div align="center">WebUI created by <a href="https://github.com/SUC-DriverOld">Sucial</a>: <a href="https://github.com/SUC-DriverOld/MedleyVox-Inference-WebUI">Githubl</a> | <a href="https://github.com/SUC-DriverOld/MedleyVox-Inference-WebUI/blob/master/LICENSE">LICENSE</a></div>''')

        audio_tab.select(fn=change_to_audio_infer, outputs=[inference_audio, inference_folder])
        folder_tab.select(fn=change_to_folder_infer, outputs=[inference_audio, inference_folder])
        select_input_dir.click(select_folder, outputs=[folder_input])
        open_input_dir.click(open_folder, inputs=[folder_input])
        select_store_btn.click(select_folder, outputs=[store_dir])
        open_store_btn.click(open_folder, inputs=[store_dir])
        inference_audio.click(inference_audio_fn, inputs=[model_name, use_overlapadd, use_gpu, separate_storage, output_format, vad_method, spectral_features, ola_window_len, ola_hop_len, w2v_nth_layer_output, use_ema_model, mix_consistent_out, reorder_chunks, skip_error, audio_input, store_dir], outputs=[output_message])
        inference_folder.click(inference_folder_fn, inputs=[model_name, use_overlapadd, use_gpu, separate_storage, output_format, vad_method, spectral_features, ola_window_len, ola_hop_len, w2v_nth_layer_output, use_ema_model, mix_consistent_out, reorder_chunks, skip_error, folder_input, store_dir], outputs=[output_message])
        stop_inference.click(stop_inference_fn, inputs=[], outputs=[output_message])
    return demo

if __name__ == "__main__":
    import argparse
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="WebUI for MedleyVox", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=60))
    parser.add_argument("-i", "--ip_address", type=str, default=None, help="IP address to run the server")
    parser.add_argument("-p", "--port", type=int, default=None, help="Port to run the server")
    parser.add_argument("-s", "--share", action="store_true", help="Enable gradio share link")
    parser.add_argument("--auto_clean_cache", action="store_true", help="Auto clean WebUI cache")
    args = parser.parse_args()

    os.makedirs("inputs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("cache", exist_ok=True)

    if args.auto_clean_cache:
        shutil.rmtree("cache", ignore_errors=True)
        os.makedirs("cache", exist_ok=True)

    os.environ["GRADIO_TEMP_DIR"] = os.path.abspath("cache/")
    os.environ["HF_HOME"] = os.path.abspath(PRETRAINED_MODEL_DIR)

    webui().queue().launch(inbrowser=True, server_name=args.ip_address, server_port=args.port, share=args.share)