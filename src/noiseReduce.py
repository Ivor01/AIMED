from pathlib import Path
import subprocess
import os


def preprocess_with_ffmpeg(input_path: str) -> None:
    
    output_dir = Path.cwd()
    input_file = Path(input_path)
    output_path = str( output_dir / f"{input_file.stem}_cleaned.wav" )

    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-af", "highpass=f=100,afftdn=nf=-28,loudnorm=I=-20:LRA=7:TP=-2",
        "-ac", "1",
        "-ar", "16000",
        output_path,
    ]
    subprocess.run(cmd, check=True)

#USAGE: void preprocess_with_ffmpeg(input_path,output_path)