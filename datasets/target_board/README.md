# Target-board dataset contract

This directory contains only the versioned contract for the target-board YOLO dataset. Raw captures, staged images, labels, caches, and generated training products remain local and must not be committed.

## One-day collection target

Collect approximately:

- **150 normal positive images** (`difficulty=clear`);
- **60–80 difficult positive images** (`difficulty=difficult`), including clutter, oblique views, partial lighting changes, natural blur, and realistic distance changes;
- **30–50 negative images** (`difficulty=negative`) containing plausible rectangular distractors but no target board.

Use genuinely different physical scenes and camera poses. Do not create train/validation/test copies by changing only exposure or drawing synthetic noise over the same image.

## Capture and annotation rules

1. Ingest only each capture directory's `original.png`; never train on dashboard overlays or debug products.
2. Positive images have exactly one YOLO label for class `0` (`target_board`). Draw a tight box around the **complete physical target board**, not an inner rectangle, printed pattern, light spot, or background frame.
3. Keep the complete board inside the image sufficiently for the later expanded-ROI corner refinement. If the board is seriously cut off, recapture it or classify it as a difficult evaluation example rather than pretending the visible fragment is a complete board.
4. Negative images have a corresponding empty `.txt` label file.
5. Every image must appear exactly once in `split-manifest.csv`, whose columns are exactly:

   ```csv
   image,scene,split,difficulty
   ```

6. `difficulty` is one of `clear`, `difficult`, or `negative`. Non-empty labels require `clear` or `difficult`; empty labels require `negative`.

## Directory layout

```text
datasets/target_board/
├── dataset.yaml
├── split-manifest.csv
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Image and label relative paths must match within their split, for example `images/train/desk-left-000001.png` and `labels/train/desk-left-000001.txt`. Manifest `image` values use the image path relative to the split directory, normally just the deterministic filename.

## Scene-based split

Assign whole scenes—not individual frames—to an approximate **70/20/10 train/validation/test split**. A scene identifier must appear in only one split. This prevents near-identical frames from the same background, lighting setup, or camera placement from leaking into evaluation.

A practical workflow is:

```powershell
python tools/prepare_target_dataset.py CAPTURE_ROOT `
  --output datasets/target_board/staging `
  --scene desk-left `
  --split train `
  --difficulty clear

python tools/validate_target_dataset.py datasets/target_board
```

Copy the staged originals into the chosen `images/<split>` directory, create matching labels, append the emitted rows to `split-manifest.csv`, and validate before training. The validator rejects unreadable images, missing or malformed labels, invalid normalized boxes, manifest/difficulty mismatches, scene leakage, and duplicate image content across splits.
