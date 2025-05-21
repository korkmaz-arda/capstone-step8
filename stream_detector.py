import time
import argparse
import cv2
import os
import sys
from pathlib import Path
import torch
import numpy as np
from ultralytics import YOLO

from utils.detect import (
    detect_trays, detect_food,
    draw_trays, draw_food,
    track_lucas_kanade
)
from utils.bbox import bbox2poly

script_dir = os.path.dirname(os.path.abspath(__file__))


def load_models(tray_path, food_path, device):
    if not os.path.isabs(tray_path):
        tray_path = os.path.join(script_dir, tray_path)
    if not os.path.isabs(food_path):
        food_path = os.path.join(script_dir, food_path)
    print(f"Loading tray model from {tray_path}")
    tray = YOLO(str(tray_path)).to(device)
    print(f"Loading food model from {food_path}")
    food = YOLO(str(food_path)).to(device)
    return tray, food


def parse_args():
    p = argparse.ArgumentParser("stream_detector")
    p.add_argument("--mode", choices=["webcam", "stream"], required=True)
    p.add_argument("--camera-index", type=int, default=2)
    p.add_argument("--stream-url", type=str, default=None)
    p.add_argument("--output-mode", choices=["display", "verbose"], default="display")
    p.add_argument("--save-display", action="store_true")
    p.add_argument("--video-out", type=str, default="./outputs/detections.mp4")
    p.add_argument("--save-log", action="store_true")
    p.add_argument("--log-out", type=str, default="detections.log")
    p.add_argument("--tray-conf", type=float, default=0.77)
    p.add_argument("--food-conf", type=float, default=0.30)
    p.add_argument("--intersection-thresh", type=float, default=0.5)
    p.add_argument("--skip-frames", type=int, default=8)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height
    union_area = area1 + area2 - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    return iou


def match_boxes(prev_boxes, tracked_boxes, threshold=0.3):
    if not prev_boxes or not tracked_boxes:
        return []
    matches = []
    used_tracked = set()
    for i, prev_box in enumerate(prev_boxes):
        best_match = -1
        best_iou = threshold
        for j, tracked_box in enumerate(tracked_boxes):
            if j in used_tracked:
                continue
            iou = calculate_iou(prev_box, tracked_box)
            if iou > best_iou:
                best_iou = iou
                best_match = j
        if best_match != -1:
            matches.append((i, best_match))
            used_tracked.add(best_match)
    return matches


def main():
    try:
        args = parse_args()
        print(f"Arguments parsed successfully")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        base = os.path.join(script_dir, "models")
        base = Path(base)
        print(f"Models directory: {base}")
        tray_path = base / "tray_detector.pt"
        food_path = base / "yolo11n.pt"
        if not os.path.exists(tray_path):
            print(f"ERROR: Tray model not found at {tray_path}")
            return
        if not os.path.exists(food_path):
            print(f"ERROR: Food model not found at {food_path}")
            return
        tray_model, food_model = load_models(tray_path, food_path, device)
        video_out_dir = os.path.dirname(args.video_out)
        if args.save_display and video_out_dir and not os.path.exists(video_out_dir):
            os.makedirs(video_out_dir, exist_ok=True)
            print(f"Created output directory: {video_out_dir}")
        if args.mode == "webcam":
            cap = cv2.VideoCapture(args.camera_index)
        else:
            if not args.stream_url:
                print("Error: --stream-url required for stream mode")
                return
            cap = cv2.VideoCapture(args.stream_url)
        if not cap.isOpened():
            print(f"❌ Could not open source: {args.mode}")
            return
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        writer = None
        if args.save_display:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.video_out, fourcc, fps, (width, height))
            print(f"✔ Will save processed video to {args.video_out}")
        log_f = None
        if args.save_log:
            log_f = open(args.log_out, "w")
            print(f"✔ Will save detection log to {args.log_out}")
        prev_gray = None
        prev_trays = []
        prev_foods = []
        frame_idx = 0
        start_t = time.time()
        cur_sec = 0
        sec_dets = []
        print("Starting stream detection. Press Ctrl+C to exit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️  Empty frame, retrying...")
                time.sleep(0.05)
                continue
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if frame_idx % args.skip_frames == 0 or prev_gray is None:
                trays = detect_trays(frame, tray_model, args.tray_conf)
                frame = draw_trays(frame, trays)
                foods = detect_food(
                    frame, food_model, args.food_conf,
                    ["banana", "apple", "sandwich", "orange", "broccoli",
                     "carrot", "hot dog", "pizza", "donut", "cake"]
                )
                frame = draw_food(frame, foods, trays, args.intersection_thresh)
                prev_gray = frame_gray.copy()
                prev_trays = trays
                prev_foods = foods
            else:
                tray_boxes = [xyxy for _, xyxy, _ in prev_trays]
                tracked_trays = track_lucas_kanade(prev_gray, frame_gray, tray_boxes)
                tray_matches = match_boxes(tray_boxes, tracked_trays)
                new_trays = []
                for prev_idx, tracked_idx in tray_matches:
                    _, _, conf = prev_trays[prev_idx]
                    tracked_box = tracked_trays[tracked_idx]
                    poly = bbox2poly(tracked_box)
                    new_trays.append((poly, tracked_box, conf))
                matched_prev = [p for p, _ in tray_matches]
                for i in range(len(prev_trays)):
                    if i not in matched_prev:
                        poly, box, conf = prev_trays[i]
                        new_conf = max(0.5, conf * 0.9)
                        new_trays.append((poly, box, new_conf))
                frame = draw_trays(frame, new_trays)
                food_boxes = [b for b, _, _ in prev_foods]
                tracked_foods = track_lucas_kanade(prev_gray, frame_gray, food_boxes)
                food_matches = match_boxes(food_boxes, tracked_foods, threshold=0.2)
                new_foods = []
                for prev_idx, tracked_idx in food_matches:
                    tracked_box = tracked_foods[tracked_idx]
                    _, conf, cls = prev_foods[prev_idx]
                    new_foods.append((tracked_box, conf, cls))
                matched_prev_food = [p for p, _ in food_matches]
                for i in range(len(prev_foods)):
                    if i not in matched_prev_food:
                        box, conf, cls = prev_foods[i]
                        new_conf = max(0.5, conf * 0.8)
                        new_foods.append((box, new_conf, cls))
                frame = draw_food(frame, new_foods, new_trays, args.intersection_thresh)
                prev_gray = frame_gray.copy()
                prev_trays = new_trays
                prev_foods = new_foods
            if writer:
                writer.write(frame)
            elapsed = time.time() - start_t
            sec = int(elapsed)
            if args.output_mode == "display":
                cv2.imshow("Stream Detector", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                for box, conf, cls in prev_foods:
                    bbox = [int(v) for v in box]
                    sec_dets.append((cls, bbox))
                if sec > cur_sec:
                    header = f"\nSecond {cur_sec}:"
                    print(header)
                    if log_f: log_f.write(header + "\n")
                    for idx, (cls, bbox) in enumerate(sec_dets, 1):
                        line = f" {idx}. {cls} {bbox}"
                        print(line)
                        if log_f: log_f.write(line + "\n")
                    cur_sec += 1
                    sec_dets = []
            frame_idx += 1
            time.sleep(0.001)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'cap' in locals() and cap is not None:
            cap.release()
        if 'writer' in locals() and writer is not None:
            writer.release()
        if args.output_mode == "display" and cv2 is not None:
            cv2.destroyAllWindows()
        if 'log_f' in locals() and log_f is not None:
            log_f.close()
        print("\n🛑 Exited.")


if __name__ == "__main__":
    main()
