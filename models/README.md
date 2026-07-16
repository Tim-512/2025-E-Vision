# Target-board model artifacts

Install training tools with `pip install -e .[training]`; Ultralytics is not a runtime dependency.

- `target-board.pt`: selected best checkpoint trained and validated on Windows.
- `target-board.onnx`: portable deployment fallback, exported and validated before TensorRT.
- `target-board.engine`: TensorRT engine built locally on the target Jetson. Never copy an engine from Windows or another Jetson.

Typical commands:

```powershell
python tools/train_target_detector.py --data datasets/target_board/dataset.yaml --model yolo11n.pt --epochs 80 --imgsz 640 --device 0 --project runs/target-board --name target-board-v1
python tools/evaluate_target_detector.py --model runs/target-board/target-board-v1/weights/best.pt --data datasets/target_board/dataset.yaml --manifest datasets/target_board/split-manifest.csv --output runs/target-board/target-board-v1/test-metrics.json
python tools/export_target_detector.py --model models/target-board.pt --format onnx --imgsz 640 --device 0
python tools/inspect_target_predictions.py --model models/target-board.pt --source datasets/target_board/images/test --labels datasets/target_board/labels/test --manifest datasets/target_board/split-manifest.csv --output runs/target-board/review-test
```

After validating the FP32 checkpoint and ONNX fallback, build TensorRT on the target Jetson:

```bash
python tools/export_target_detector.py --model models/target-board.pt --format engine --imgsz 640 --device 0
```

For each selected release, record all of the following:

```text
release and calibration/test date:
model source checkpoint:
dataset split-manifest.csv SHA-256:
training command:
metrics JSON path and precision/recall/mAP50/mAP50-95:
clear_recall and difficult_recall:
negative_false_positive_images and max_negative_high_confidence_streak:
ONNX export command and Jetson engine export command:
target-board.pt / target-board.onnx / target-board.engine SHA-256:
Windows training environment:
Jetson model, JetPack, CUDA, TensorRT, and Ultralytics versions:
notes:
```

Use `Get-FileHash -Algorithm SHA256` on Windows or `sha256sum` on Jetson. If Jetson-local engine export or validation fails, keep the tested ONNX fallback instead of deploying an unverified engine.
