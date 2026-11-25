"""E7: COCO and VisDrone YOLO11m ensemble for human detection."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion

ROOT = Path(__file__).resolve().parent


def predict(model, path, config, device):
    result = model.predict(source=str(path), imgsz=config['image_size'],
                           conf=0.01, iou=config['nms_iou'], classes=[0],
                           max_det=config['max_det'], device=device,
                           verbose=False)[0]
    return list(zip(result.boxes.conf.cpu().tolist(),
                    result.boxes.xywhn.cpu().tolist()))


def xywh_to_xyxy(box):
    x, y, w, h = box
    return np.clip([x-w/2, y-h/2, x+w/2, y+h/2], 0, 1).tolist()


def fuse(first, second, config):
    groups = [[(s, b) for s, b in first if s >= config['pretrained_confidence']],
              [(s, b) for s, b in second if s >= config['finetuned_confidence']]]
    boxes = [[xywh_to_xyxy(b) for s, b in group] for group in groups]
    scores = [[s for s, b in group] for group in groups]
    labels = [[0]*len(group) for group in groups]
    fused_boxes, fused_scores, _ = weighted_boxes_fusion(
        boxes, scores, labels, weights=config['model_weights'],
        iou_thr=config['fusion_iou'], skip_box_thr=0.0, conf_type='avg')
    parts = []
    for (x1, y1, x2, y2), score in zip(fused_boxes, fused_scores):
        if score >= config['output_confidence']:
            parts.append(f'1 {(x1+x2)/2:.6f} {(y1+y2)/2:.6f} {x2-x1:.6f} {y2-y1:.6f}')
    return ' '.join(parts) if parts else ' '


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--sample', type=Path, help='Optional sample CSV for image order')
    parser.add_argument('--output', type=Path, default=Path('submission.csv'))
    parser.add_argument('--device', default=None, help='0 for GPU or cpu')
    args = parser.parse_args()
    config = json.loads((ROOT/'final_config.json').read_text())
    models = [YOLO(str(ROOT/'weights'/config[key])) for key in ('pretrained_model', 'finetuned_model')]
    paths = sorted(p for p in args.images.rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png'})
    image_map = {p.stem: p for p in paths}
    if len(image_map) != len(paths):
        raise ValueError('Duplicate image IDs')
    ids = pd.read_csv(args.sample)['image_id'].tolist() if args.sample else list(image_map)
    if not ids:
        raise ValueError('No images found')
    missing = set(ids)-image_map.keys()
    if missing:
        raise ValueError(f'Missing images: {sorted(missing)[:10]}')
    rows = []
    for i, image_id in enumerate(ids, 1):
        first, second = [predict(m, image_map[image_id], config, args.device) for m in models]
        rows.append((image_id, fuse(first, second, config)))
        if i % 25 == 0 or i == len(ids):
            print(f'Processed {i}/{len(ids)}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=['image_id','PredictionString']).to_csv(args.output, index=False)
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
