#!/usr/bin/env python3
"""
Merge raw transcripts into lecture-by-lecture markdown files.

Groups transcripts by lecture number (Part1.1, Part1.2 → Lecture 1, etc.)
and creates clean markdown files ready for textbook editing.
"""

import re
from pathlib import Path
from collections import defaultdict

RAW_DIR = Path("/Users/rpillai/Library/CloudStorage/Dropbox/PERSONAL-PROJECTS/heat-transfer-textbook/transcripts/raw")
OUTPUT_DIR = Path("/Users/rpillai/Library/CloudStorage/Dropbox/PERSONAL-PROJECTS/heat-transfer-textbook/transcripts/by-lecture")

# Mapping of lecture numbers to topics (from 2021 course structure)
LECTURE_TOPICS = {
    1: "Fundamentals of Heat Transfer",
    2: "Boundary Layers and Governing Equations",
    3: "External Convection and Correlations",
    4: "Internal Convection (Pipe Flows)",
    5: "Heat Exchangers",
    6: "Conduction Fundamentals",
    7: "Extended Surfaces (Fins)",
    8: "Advanced Conduction and Transient",
    9: "Radiation Properties",
    10: "Radiation Exchange",
    11: "Combined Heat Transfer Modes",
    12: "Phase Change and Special Topics",
}

def parse_filename(filename: str) -> tuple[int, int] | None:
    """
    Extract lecture and part number from filename.

    Examples:
        Part1.1.txt → (1, 1)
        Part2.6.txt → (2, 6)
        Part 10.3.txt → (10, 3)
        W1.1.txt → (1, 1) for worked examples
    """
    # Match patterns like "Part1.1", "Part 10.3", "Part10.0"
    match = re.match(r'Part\s*(\d+)\.(\d+)', filename, re.IGNORECASE)
    if match:
        return int(match.group(1)), int(match.group(2))

    # Match patterns like "W1.1" (worked examples - map to lecture 1)
    match = re.match(r'W(\d+)\.(\d+)', filename, re.IGNORECASE)
    if match:
        # W1, W2 are supplementary to lecture 1
        return 1, 100 + int(match.group(2))  # Put at end

    return None

def read_transcript(path: Path) -> str:
    """Read transcript file, strip header comments."""
    text = path.read_text(encoding='utf-8')
    lines = text.split('\n')

    # Skip header lines starting with #
    content_lines = []
    for line in lines:
        if line.startswith('#'):
            continue
        content_lines.append(line)

    return '\n'.join(content_lines).strip()

def main():
    print("=" * 60)
    print("Merging Transcripts by Lecture")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all .txt transcripts
    transcripts = list(RAW_DIR.glob("*.txt"))
    print(f"Found {len(transcripts)} transcript files\n")

    # Group by lecture number
    by_lecture: dict[int, list[tuple[int, Path]]] = defaultdict(list)
    unmatched = []

    for path in transcripts:
        parsed = parse_filename(path.name)
        if parsed:
            lecture_num, part_num = parsed
            by_lecture[lecture_num].append((part_num, path))
        else:
            unmatched.append(path)

    # Report unmatched files
    if unmatched:
        print(f"Unmatched files (will be skipped):")
        for p in unmatched:
            print(f"  - {p.name}")
        print()

    # Process each lecture
    for lecture_num in sorted(by_lecture.keys()):
        parts = by_lecture[lecture_num]
        parts.sort(key=lambda x: x[0])  # Sort by part number

        topic = LECTURE_TOPICS.get(lecture_num, "Unknown Topic")
        output_path = OUTPUT_DIR / f"lecture-{lecture_num:02d}.md"

        print(f"Lecture {lecture_num}: {topic}")
        print(f"  Parts: {len(parts)}")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# Lecture {lecture_num}: {topic}\n\n")
            f.write(f"*Auto-generated transcript - requires editing*\n\n")
            f.write("---\n\n")

            for part_num, path in parts:
                content = read_transcript(path)
                if not content:
                    continue

                # Section header
                if part_num < 100:
                    f.write(f"## Part {lecture_num}.{part_num}\n\n")
                else:
                    f.write(f"## Worked Example {part_num - 100}\n\n")

                f.write(content)
                f.write("\n\n---\n\n")

        print(f"  Saved: {output_path.name}")

    print()
    print("=" * 60)
    print(f"Created {len(by_lecture)} lecture files in:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
