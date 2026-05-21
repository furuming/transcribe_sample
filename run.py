from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from faster_whisper import WhisperModel
from huggingface_hub import login


MODEL_SIZE = "tiny"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe an audio file with Faster-Whisper tiny on CPU."
    )
    parser.add_argument(
        "audio_file",
        type=Path,
        help="Path to the audio file to transcribe.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for JSON output. Defaults to output/.",
    )
    return parser.parse_args()


def login_to_hugging_face_if_available() -> bool:
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token:
        return False

    login(token=token)
    return True


def segment_to_dict(segment: Any) -> dict[str, Any]:
    return {
        "id": segment.id,
        "seek": segment.seek,
        "start": segment.start,
        "end": segment.end,
        "text": segment.text.strip(),
        "temperature": segment.temperature,
        "avg_logprob": segment.avg_logprob,
        "compression_ratio": segment.compression_ratio,
        "no_speech_prob": segment.no_speech_prob,
        "words": [
            {
                "start": word.start,
                "end": word.end,
                "word": word.word.strip(),
                "probability": word.probability,
            }
            for word in (segment.words or [])
        ],
    }


def build_output_path(audio_file: Path, output_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"{audio_file.stem}_{timestamp}.json"


def transcribe(audio_file: Path) -> dict[str, Any]:
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    segments_iter, info = model.transcribe(
        str(audio_file),
        beam_size=5,
        word_timestamps=True,
    )
    segments = [segment_to_dict(segment) for segment in segments_iter]

    return {
        "source": {
            "path": str(audio_file),
            "file_name": audio_file.name,
        },
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": {
            "library": "faster-whisper",
            "size": MODEL_SIZE,
            "device": DEVICE,
            "compute_type": COMPUTE_TYPE,
        },
        "language": {
            "code": info.language,
            "probability": info.language_probability,
        },
        "duration": info.duration,
        "duration_after_vad": info.duration_after_vad,
        "text": " ".join(segment["text"] for segment in segments).strip(),
        "segments": segments,
    }


def main() -> None:
    args = parse_args()
    audio_file = args.audio_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not audio_file.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    logged_in = login_to_hugging_face_if_available()
    result = transcribe(audio_file)
    result["hugging_face"] = {"logged_in": logged_in}

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = build_output_path(audio_file, output_dir)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(output_path)


if __name__ == "__main__":
    main()
