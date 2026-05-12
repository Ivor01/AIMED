import warnings

warnings.filterwarnings(
    "ignore",
    message="torchcodec is not installed correctly.*"
)

warnings.filterwarnings(
    "ignore",
    message="TensorFloat-32.*"
)

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"pyannote\.audio\.core\.io"
)

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"torch\.backends\.cuda\.__init__"
)

warnings.filterwarnings(
    "ignore",
    message=r".*degrees of freedom is <= 0.*",
    category=UserWarning
)

warnings.filterwarnings(
    "ignore",
    message=r".*Please use the new API settings to control TF32 behavior.*",
    category=UserWarning
)


import os
import json
import argparse
import numpy as np
import soundfile as sf
import torch
import torchaudio
from pyannote.audio import Pipeline


class SpeakerDiarizer:
    def __init__(self, hf_token: str, device: str = "auto"):
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token
        )

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.pipeline.to(self.device)

    def load_audio_without_torchcodec(self, audio_path: str, target_sr: int = 16000):
        audio, sr = sf.read(audio_path, dtype="float32")

        # Ako je stereo, pretvori u mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        waveform = torch.from_numpy(audio).unsqueeze(0)

        # Resample ako treba
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=target_sr
            )
            waveform = resampler(waveform)
            sr = target_sr

        return waveform, sr

    def diarize(self, audio_path: str):
        waveform, sample_rate = self.load_audio_without_torchcodec(audio_path)

        file = {
            "waveform": waveform,
            "sample_rate": sample_rate
        }

        diarization = self.pipeline(file)
        

        segments = []

        annotation = diarization.exclusive_speaker_diarization

        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segments.append({
                "start": round(float(turn.start), 3),
                "end": round(float(turn.end), 3),
                "speaker": speaker
            })

        return segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path")
    parser.add_argument("--output", default="diarization_segments.json")
    args = parser.parse_args()

    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise RuntimeError("HF_TOKEN nije postavljen.")

    diarizer = SpeakerDiarizer(hf_token=hf_token)
    segments = diarizer.diarize(args.audio_path)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    print(f"Spremljeno {len(segments)} segmenata u {args.output}")


if __name__ == "__main__":
    main()