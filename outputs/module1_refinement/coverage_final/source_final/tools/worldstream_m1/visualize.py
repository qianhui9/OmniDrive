import argparse
from pathlib import Path
import json
import cv2
import numpy as np


def draw_frame(frame, result, path, current_regions=None):
    panels = []
    for cid, camera in frame.cameras.items():
        image = camera.image.copy()
        # Conservative invalidation red tiles; independent of semantic detections.
        regions = result.invalidation_regions[cid]
        overlay = image.copy()
        for r in regions['regions']:
            x1, y1, x2, y2 = map(int, r)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 30, 30), -1)
        image = cv2.addWeighted(image, 0.8, overlay, 0.2, 0)
        for aid, check in result.diagnostics.get('anchors', {}).items():
            c = check['cameras'][cid]
            old_region = c.get('historical_region')
            if old_region:
                x1, y1, x2, y2 = map(int, old_region)
                cv2.rectangle(image, (x1, y1), (x2, y2), (80, 140, 255), 1)
            r = c.get('predicted_region')
            if r:
                x1, y1, x2, y2 = map(int, r)
                color = (0, 255, 80) if check['valid'] else (255, 180, 0)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
                cv2.putText(image, aid.split(':')[-1], (x1, max(12, y1)), cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1)
                if check.get('anchor_type') == 'map':
                    uv = np.asarray(c.get('projected_points', []))
                    if uv.size and np.isfinite(uv).all():
                        cv2.polylines(image, [np.round(uv).astype(np.int32)], False, color, 2)
        for uv in frame.metadata.get('current_map_polylines', {}).get(cid, []):
            cv2.polylines(image, [np.round(uv).astype(np.int32)], False, (220, 80, 255), 2)
        for r in (current_regions or {}).get(cid, []):
            x1, y1, x2, y2 = map(int, r)
            cv2.rectangle(image, (x1, y1), (x2, y2), (220, 80, 255), 1)
        for r in regions['changed_regions']:
            x1, y1, x2, y2 = map(int, r)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 40, 40), 2)
        cv2.putText(image, cid, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
        # Different camera resolutions are supported; pad each panel for a contact sheet.
        panels.append(image)
    if not panels: return
    h, w = max(i.shape[0] for i in panels), max(i.shape[1] for i in panels)
    panels = [cv2.copyMakeBorder(i, 0, h-i.shape[0], 0, w-i.shape[1], cv2.BORDER_CONSTANT) for i in panels]
    while len(panels) % 3: panels.append(np.zeros_like(panels[0]))
    rows = [np.concatenate(panels[i:i+3], axis=1) for i in range(0, len(panels), 3)]
    canvas = np.concatenate(rows, axis=0)
    banner = np.zeros((65, canvas.shape[1], 3), dtype=np.uint8)
    legend = ['Blue: historical crop | Green: certified | Orange: predicted invalid | Magenta: current full output',
              'Red tint: unexplained/invalidation; red box: local change. No new semantic detection claimed.']
    for j, text in enumerate(legend):
        cv2.putText(banner, text, (8, 23 + 25*j), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
    canvas = np.concatenate([banner, canvas])
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser(description='Rebuild a contact sheet from saved M1 replay state')
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    root = Path(args.run_dir)
    images = sorted((root / 'visualizations').glob('*.jpg'))
    if not images: raise SystemExit('No rendered frames. Run run_streaming with visualization enabled.')
    thumbs = []
    for path in images:
        image = cv2.imread(str(path))
        thumbs.append(cv2.resize(image, (960, int(image.shape[0] * 960 / image.shape[1]))))
    cv2.imwrite(str(root / 'contact_sheet.jpg'), np.concatenate(thumbs, axis=0))
    print(root / 'contact_sheet.jpg')


if __name__ == '__main__': main()
