import torch
import runpod
from srt import compose, Subtitle
from datetime import timedelta
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
)
from os.path import dirname


def handler(job):
    job_input = job["input"]
    video_url = job_input["url"]
    output_fmt = ""
    if "output" in job_input:
        output_fmt = job_input["output"]

    safetensors_path = f"{dirname(__file__)}/model/"
    device_id = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        safetensors_path, torch_dtype=torch_dtype, use_safetensors=True
    )
    model.to(device_id)

    processor = AutoProcessor.from_pretrained(safetensors_path)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        torch_dtype=torch_dtype,
        device=device_id,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
    )

    pipe.model = pipe.model.to_bettertransformer()

    outputs = pipe(
        video_url,
        chunk_length_s=30,
        batch_size=24,
        return_timestamps=True,
    )

    if output_fmt == "srt":
        subtitles = []
        for idx, chunk in enumerate(outputs["chunks"]):
            ts = chunk["timestamp"]
            content = chunk["text"]

            subtitles.append(
                Subtitle(
                    index=idx,
                    start=timedelta(seconds=ts[0]),
                    end=timedelta(seconds=ts[1]),
                    content=content,
                )
            )

        return compose(subtitles)

    return outputs["text"]


runpod.serverless.start({"handler": handler})
