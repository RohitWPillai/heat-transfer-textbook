# Heat Transfer Textbook - Project Status

## Quick Summary

**Goal**: Open-source interactive heat transfer textbook with Python, targeting JOSE publication.

**Structure**: 4 parts, 19 chapters (Intro → Convection → Conduction → Radiation)

## What's Done

| Component | Status | Location |
|-----------|--------|----------|
| Quarto skeleton | ✅ Complete | `_quarto.yml`, chapter `.qmd` files |
| Chapter stubs (19) | ✅ With equations + code | `part-1/2/3/4-*/` |
| Python utilities | ✅ Core functions | `src/` (properties, correlations, conduction, radiation) |
| Book builds | ✅ HTML output | `_book/index.html` |
| Video transcription | 🔄 86/109 | `transcripts/raw/` |

## Content Sources

| Source | Location | Use |
|--------|----------|-----|
| Video transcripts | `transcripts/raw/*.txt` | Primary text (your words) |
| 2021 Lecture slides | `Teaching/Thermofluids/2021/LECTURES/` | Equations, structure |
| 2024 In-Person Sessions | `Teaching/Thermofluids/2024/IN-PERSON SESSIONS/` | Complex worked examples |
| Tutorial LaTeX | `2021/TUTORIALS/*-LaTeX/` | Problem sets |
| HT book ideas | Obsidian `Apple Notes/HT book.md` | Real-world examples |

## Key Examples to Include

| Example | Chapter | Why Memorable |
|---------|---------|---------------|
| Turkish sand coffee | Convection | Unusual heat source |
| Buffon's Age of Earth | Transient | Historical + physics |
| Thermal paste in cooking | Contact resistance | Everyday application |
| Wet bulb temperature | Phase change | Climate relevance |
| Xiao long bao | Phase change | Food science |
| Murano glass making | Thermal shock | Craft + engineering |

## Copyright Strategy

- ✅ Text: From your transcripts (your words)
- ✅ Equations: Scientific facts (not copyrightable)
- ✅ Correlations: Cite original papers, not Incropera
- ✅ Figures: All Python-generated
- ✅ Properties: CoolProp/NIST (open source)
- ✅ Examples: Original contexts (cooking, climate, etc.)

## Next Steps

1. [ ] Complete transcription (23 videos remaining)
2. [ ] Run `merge_transcripts.py` to group by lecture
3. [ ] Edit transcripts: spoken → written style
4. [ ] Fill chapter stubs with transcript content
5. [ ] Add Python visualizations
6. [ ] Create original worked examples
7. [ ] Peer review
8. [ ] JOSE submission

## File Structure

```
heat-transfer-textbook/
├── _quarto.yml           # Book config
├── index.qmd             # Preface
├── part-1-introduction/  # Ch 1-3
├── part-2-convection/    # Ch 4-9
├── part-3-conduction/    # Ch 10-14
├── part-4-radiation/     # Ch 15-19
├── src/                  # Python utilities
├── transcripts/
│   ├── raw/              # Whisper output
│   └── by-lecture/       # Merged (after processing)
├── BOOK_PLAN.md          # Detailed structure
├── PROJECT_STATUS.md     # This file
└── source-materials-summary.md
```

## Commands

```bash
# Preview book with live reload
quarto preview

# Build HTML
quarto render

# Check transcription progress
ls transcripts/raw/*.txt | wc -l

# Restart transcription (if interrupted)
python3 transcribe_videos.py

# Merge transcripts by lecture
python3 merge_transcripts.py
```
