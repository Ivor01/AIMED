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
from typing import Optional

class SpeakerDiarizer:
    def __init__(
        self,
        hf_token: Optional[str] = None,
        device: str = "auto"
    ):
        self.hf_token = hf_token or os.getenv("HF_TOKEN")

        if not self.hf_token:
            raise RuntimeError(
                "HF_TOKEN is not set. Set it as an environment variable "
                "or pass hf_token directly to SpeakerDiarizer."
            )
        
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
    
    def diarize_to_file(self, audio_path: str, output_path: str):
        segments = self.diarize(audio_path)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)

        return segments


def main():
    parser = argparse.ArgumentParser(
        description="Run speaker diarization using pyannote.audio."
    )

    parser.add_argument(
        "audio_path",
        help="Path to input audio file, e.g. processed.wav"
    )

    parser.add_argument(
        "--output",
        default="diarization_segments.json",
        help="Path to output JSON file"
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use"
    )
    
    args = parser.parse_args()

    diarizer = SpeakerDiarizer(
        device=args.device
    )

    segments = diarizer.diarize_to_file(
        audio_path=args.audio_path,
        output_path=args.output
    )

    print(f"Saved {len(segments)} segments to {args.output}")

if __name__ == "__main__":
    main()


#korištenje u terminalu python diar_test.py processed.wav
# možemo dodati: --output filepath, --device cpu/gpu i kasnije broj govornika
'''Iz drugog programa: importamo klasu SpeakerDiarizer
    diarizer = SpeakerDiarizer(output=...,device=...,min_speakers=...)
    segments = diarizer.diarize(audio_file)
'''