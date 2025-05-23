import time
import argparse
import csv
import tempfile
import os
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO

from utils.detect import (
    detect_trays, detect_food,
    draw_trays, draw_food,
    track_lucas_kanade
)
from utils.bbox import bbox2poly


def load_models(tray_path, food_path, device):
    print(f"Loading tray model from: {tray_path}")
    tray_model = YOLO(str(tray_path)).to(device)
    print(f"Loading food model from: {food_path}")
    food_model = YOLO(str(food_path)).to(device)
    return tray_model, food_model


def detect_optical_flow(
    video_path, tray_model, food_model,
    tray_conf, food_conf, skip_frames,
    save_result=False, output_path=None
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    if save_result:
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    else:
        tmp_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        writer = cv2.VideoWriter(tmp_output.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    prev_gray = None
    prev_trays = []
    prev_foods = []
    frame_idx = 0
    food_classes = [
        "banana", "apple", "sandwich", "orange", "broccoli",
        "carrot", "hot dog", "pizza", "donut", "cake"
    ]
    intersection_threshold = 0.5

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame_idx % skip_frames == 0 or prev_gray is None:
            trays = detect_trays(frame, tray_model, tray_conf)
            frame = draw_trays(frame, trays)
            foods = detect_food(frame, food_model, food_conf, food_classes)
            frame = draw_food(frame, foods, trays, intersection_threshold)
            prev_trays = trays
            prev_foods = foods
        else:
            tray_boxes = [xyxy for _, xyxy, _ in prev_trays]
            tracked_trays = track_lucas_kanade(prev_gray, frame_gray, tray_boxes)
            new_trays = [(bbox2poly(box), box, 0.75) for box in tracked_trays]
            frame = draw_trays(frame, new_trays)

            food_boxes = [xyxy for xyxy, _, _ in prev_foods]
            tracked_foods = track_lucas_kanade(prev_gray, frame_gray, food_boxes)
            new_foods = [
                (tracked_box, conf, cls)
                for tracked_box, (_, conf, cls) in zip(tracked_foods, prev_foods)
            ]
            frame = draw_food(frame, new_foods, new_trays, intersection_threshold)

            prev_trays = new_trays
            prev_foods = new_foods

        writer.write(frame)
        prev_gray = frame_gray.copy()
        frame_idx += 1

    cap.release()
    writer.release()
    if not save_result:
        os.remove(tmp_output.name)

    elapsed = time.time() - start_time
    return round(elapsed, 2)


def run_benchmark(video_dir, skip_values, model_dir, tray_conf, food_conf, output_csv, save_results, output_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"DEVICE: {device}")
    tray_model_path = Path(model_dir) / "tray_detector.pt"
    food_model_path = Path(model_dir) / "yolo11n.pt"
    tray_model, food_model = load_models(tray_model_path, food_model_path, device)

    video_dir = Path(video_dir)
    videos = sorted([f for f in video_dir.glob("*.mp4")])
    results = []

    if save_results:
        output_dir = Path(output_dir or "benchmark-outputs")
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📁 Found {len(videos)} videos in {video_dir}")
    print(f"▶ Benchmarking with skip frame values: {skip_values}\n")

    for video in videos:
        row = [video.name]
        print(f"⏳ {video.name}")
        stem = video.stem
        suffix = video.suffix

        for skip in skip_values:
            print(f"  - skip={skip}", end=" ... ")
            out_path = None
            if save_results:
                out_path = output_dir / f"{stem}_{skip}{suffix}"
            duration = detect_optical_flow(
                video_path=video,
                tray_model=tray_model,
                food_model=food_model,
                tray_conf=tray_conf,
                food_conf=food_conf,
                skip_frames=skip,
                save_result=save_results,
                output_path=out_path
            )
            print(f"{duration} sec")
            row.append(duration)
        results.append(row)

    # Write CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["video"] + [str(s) for s in skip_values]
        writer.writerow(header)
        writer.writerows(results)

    print(f"\n✅ Benchmark complete. Results saved to {output_csv}")
    if save_results:
        print(f"📂 Output videos saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Benchmark optical flow (Lucas Kanade)")
    parser.add_argument("--video-dir", type=str, required=True, help="Directory containing input .mp4 videos")
    parser.add_argument("--skips", type=int, nargs="+", required=True, help="List of skip frame values")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory containing tray/food .pt models")
    parser.add_argument("--tray-conf", type=float, default=0.77)
    parser.add_argument("--food-conf", type=float, default=0.25)
    parser.add_argument("--csv", type=str, default="benchmark.csv", help="CSV output file name")
    parser.add_argument("--save-results", action="store_true", help="Save output videos instead of discarding them")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save result videos (default: benchmark-outputs/)")

    args = parser.parse_args()

    run_benchmark(
        video_dir=args.video_dir,
        skip_values=args.skips,
        model_dir=args.model_dir,
        tray_conf=args.tray_conf,
        food_conf=args.food_conf,
        output_csv=args.csv,
        save_results=args.save_results,
        output_dir=args.output_dir
    )


"""
TRY:

python3 benchmark.py \
  --video-dir ./videos/benchmark-input \
  --skips 1 3 5 8 12 16 \
  --csv benchmark.csv \
  --save-results \
  --output-dir ./videos/benchmark-output/gpu
"""