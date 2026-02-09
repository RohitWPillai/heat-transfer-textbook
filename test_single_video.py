#!/usr/bin/env python3
"""Quick test: transcribe one short video to verify setup works."""

import os
from pathlib import Path
from faster_whisper import WhisperModel

# Set HT_VIDEO_DIR environment variable to your video captures folder
VIDEO = Path(os.environ.get("HT_VIDEO_DIR", "")) / "WEEK1" / "1.0Welcome.mp4"
OUTPUT_DIR = Path(__file__).resolve().parent / "transcripts" / "raw"

print("Loading Whisper model (small)...")
model = WhisperModel("small", device="cpu", compute_type="int8")
print("Model loaded.\n")

print(f"Transcribing: {VIDEO.name}")
segments, info = model.transcribe(str(VIDEO), beam_size=5, language="en", vad_filter=True)

all_segments = list(segments)
print(f"Duration: {info.duration:.0f}s, Segments: {len(all_segments)}\n")

# Save TXT
txt_path = OUTPUT_DIR / f"{VIDEO.stem}.txt"
with open(txt_path, "w") as f:
    f.write(f"# Transcript: {VIDEO.name}\n\n")
    for seg in all_segments:
        f.write(f"{seg.text.strip()}\n\n")

print(f"Saved: {txt_path}")
print("\n--- First 500 chars of transcript ---\n")
print(txt_path.read_text()[:500])
