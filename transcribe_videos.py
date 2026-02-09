#!/usr/bin/env python3
"""
Transcribe video lectures using faster-whisper.
Outputs SRT (subtitles) and TXT (plain text) files.

This script ONLY READS video files - it never modifies them.
All outputs go to the transcripts/raw/ folder.
"""

import os
import sys
from pathlib import Path
from faster_whisper import WhisperModel

# Configuration
VIDEO_DIR = Path("/Users/rpillai/Library/CloudStorage/Dropbox/Teaching/Thermofluids/2021/CAPTURES")
OUTPUT_DIR = Path("/Users/rpillai/Library/CloudStorage/Dropbox/PERSONAL-PROJECTS/heat-transfer-textbook/transcripts/raw")

# Model size: "tiny", "base", "small", "medium", "large-v3"
# Smaller = faster but less accurate. "small" is a good balance.
MODEL_SIZE = "small"

def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def transcribe_video(model: WhisperModel, video_path: Path, output_dir: Path) -> bool:
    """Transcribe a single video file, output SRT and TXT."""

    stem = video_path.stem  # filename without extension
    srt_path = output_dir / f"{stem}.srt"
    txt_path = output_dir / f"{stem}.txt"

    # Skip if already transcribed
    if srt_path.exists() and txt_path.exists():
        print(f"  [SKIP] Already transcribed: {stem}")
        return True

    print(f"  [TRANSCRIBING] {video_path.name}...")

    try:
        segments, info = model.transcribe(
            str(video_path),
            beam_size=5,
            language="en",
            vad_filter=True,  # Filter out non-speech
        )

        # Collect segments
        all_segments = list(segments)

        # Write SRT file
        with open(srt_path, "w", encoding="utf-8") as srt_file:
            for i, seg in enumerate(all_segments, 1):
                srt_file.write(f"{i}\n")
                srt_file.write(f"{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}\n")
                srt_file.write(f"{seg.text.strip()}\n\n")

        # Write TXT file (plain text, one paragraph per segment)
        with open(txt_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(f"# Transcript: {video_path.name}\n")
            txt_file.write(f"# Duration: {info.duration:.1f} seconds\n\n")
            for seg in all_segments:
                txt_file.write(f"{seg.text.strip()}\n\n")

        print(f"  [DONE] {stem} ({info.duration:.0f}s, {len(all_segments)} segments)")
        return True

    except Exception as e:
        print(f"  [ERROR] {stem}: {e}")
        return False

def find_videos(base_dir: Path) -> list[Path]:
    """Find all MP4 files in the directory tree."""
    videos = []
    for pattern in ["*.mp4", "*.MP4"]:
        videos.extend(base_dir.rglob(pattern))
    return sorted(videos)

def main():
    print("=" * 60)
    print("Video Lecture Transcription")
    print("=" * 60)
    print(f"Video source: {VIDEO_DIR}")
    print(f"Output dir:   {OUTPUT_DIR}")
    print(f"Model:        {MODEL_SIZE}")
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find videos
    videos = find_videos(VIDEO_DIR)
    print(f"Found {len(videos)} video files")
    print()

    if not videos:
        print("No videos found!")
        return

    # Load model (downloads on first run)
    print(f"Loading Whisper model '{MODEL_SIZE}'...")
    print("(First run will download the model - this may take a few minutes)")
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    print("Model loaded.\n")

    # Transcribe each video
    success = 0
    failed = 0

    for i, video in enumerate(videos, 1):
        # Show relative path from CAPTURES folder
        rel_path = video.relative_to(VIDEO_DIR)
        print(f"[{i}/{len(videos)}] {rel_path}")

        if transcribe_video(model, video, OUTPUT_DIR):
            success += 1
        else:
            failed += 1

    print()
    print("=" * 60)
    print(f"Complete: {success} transcribed, {failed} failed")
    print(f"Outputs in: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
