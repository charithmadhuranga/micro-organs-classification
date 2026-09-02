"""Generate synthetic microorganism-like test images and a test video for e2e testing."""

import os
import math
import random
import argparse

import cv2
import numpy as np


def draw_amoeba(img, cx, cy, r, color):
    """Irregular blob with pseudopods."""
    pts = []
    for i in range(24):
        ang = math.tau * i / 24
        rr = r * random.uniform(0.7, 1.5)
        pts.append((int(cx + rr * math.cos(ang)), int(cy + rr * math.sin(ang))))
    cv2.fillPoly(img, [np.array(pts)], color)


def draw_euglena(img, cx, cy, r, color):
    """Fusiform body with a flagellum."""
    ell = (cx, cy)
    axes = (int(r * 0.5), int(r * 1.6))
    cv2.ellipse(img, ell, axes, 20, 0, 360, color, -1)
    tip_x = int(cx - axes[1] * math.cos(math.radians(20)))
    tip_y = int(cy - axes[1] * math.sin(math.radians(20)))
    cv2.line(img, (tip_x, tip_y), (tip_x - int(r), tip_y - int(r)), color, 2)


def draw_hydra(img, cx, cy, r, color):
    """Small polyp with tentacles."""
    body = (cx, cy + int(r * 1.8))
    cv2.line(img, (cx, cy + int(r * 3.2)), (cx, cy + int(r * 0.2)), color, max(3, int(r * 0.3)))
    for i in range(6):
        ang = math.radians(-60 + i * 24)
        tx = cx + int(r * 1.5 * math.cos(ang))
        ty = cy + int(r * 1.2)
        cv2.line(img, (cx, cy + int(r * 0.2)), (tx, ty), color, max(2, int(r * 0.15)))


def draw_paramecium(img, cx, cy, r, color):
    """Slipper/oval shape with cilia."""
    cv2.ellipse(img, (cx, cy), (int(r * 1.3), int(r * 0.7)), 10, 0, 360, color, -1)
    for i in range(0, 360, 20):
        ang = math.radians(i)
        x = int(cx + r * 1.05 * math.cos(ang))
        y = int(cy + r * 0.55 * math.sin(ang))
        cv2.circle(img, (x, y), max(1, int(r * 0.05)), color, -1)


def draw_rod_bacteria(img, cx, cy, r, color):
    """Rod/capsule shape."""
    cv2.ellipse(img, (cx, cy), (int(r * 1.8), int(r * 0.45)), 0, 0, 360, color, -1)


def draw_spherical_bacteria(img, cx, cy, r, color):
    """Clusters of spheres (cocci)."""
    for dx, dy, sc in [(-r, 0, 0.8), (r, 0, 0.8), (0, -r * 0.8, 0.7), (0, r * 0.8, 0.7), (0, 0, 1.0)]:
        cv2.circle(img, (int(cx + dx), int(cy + dy)), int(r * sc), color, -1)


def draw_spiral_bacteria(img, cx, cy, r, color):
    """Spiral/corkscrew shape."""
    pts = []
    for t in range(60):
        ang = math.radians(t * 12)
        x = cx + int(r * 0.4 * math.cos(ang))
        y = cy + int(t * r * 0.08) - int(r * 1.2)
        pts.append((x, y))
    cv2.polylines(img, [np.array(pts)], False, color, max(2, int(r * 0.2)))


def draw_yeast(img, cx, cy, r, color):
    """Rounded cell with budding daughter cell."""
    cv2.circle(img, (int(cx), int(cy)), int(r), color, -1)
    cv2.circle(img, (int(cx + r * 1.4), int(cy - r * 0.6)), int(r * 0.5), color, -1)


DRAWERS = {
    "Amoeba": draw_amoeba,
    "Euglena": draw_euglena,
    "Hydra": draw_hydra,
    "Paramecium": draw_paramecium,
    "Rod_bacteria": draw_rod_bacteria,
    "Spherical_bacteria": draw_spherical_bacteria,
    "Spiral_bacteria": draw_spiral_bacteria,
    "Yeast": draw_yeast,
}

COLORS = [
    (140, 220, 180), (230, 170, 120), (150, 160, 240), (200, 140, 210),
    (120, 220, 220), (240, 210, 120), (180, 160, 230), (160, 230, 210),
]


def generate_image(cls_name, size=(256, 256)):
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    img[:] = random.randint(15, 35)
    for _ in range(40):
        bx, by = random.randint(0, size[0]), random.randint(0, size[1])
        cv2.circle(img, (bx, by), random.randint(1, 3),
                   (random.randint(40, 70),) * 3, -1)
    drawer = DRAWERS[cls_name]
    color = random.choice(COLORS)
    cx, cy = size[0] // 2 + random.randint(-20, 20), size[1] // 2 + random.randint(-20, 20)
    r = random.randint(int(size[0] * 0.12), int(size[0] * 0.2))
    drawer(img, cx, cy, r, color)
    for _ in range(6):
        gx, gy = random.randint(0, size[0]), random.randint(0, size[1])
        cv2.circle(img, (gx, gy), random.randint(1, 2), (90, 130, 110), -1)
    return img


def generate_dataset(dataset_path, per_class=12, size=(256, 256)):
    for cls_name in DRAWERS:
        cls_dir = os.path.join(dataset_path, cls_name)
        os.makedirs(cls_dir, exist_ok=True)
        for i in range(per_class):
            img = generate_image(cls_name, size)
            fname = os.path.join(cls_dir, f"{cls_name}_{i:03d}.jpg")
            cv2.imwrite(fname, img)
    print(f"Generated dataset at {dataset_path} "
          f"({len(DRAWERS)} classes x {per_class} images)")


def generate_video(path, cls_names, duration=3.0, fps=10, size=(640, 480)):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, size)
    total_frames = int(duration * fps)
    for f in range(total_frames):
        cls_name = cls_names[f % len(cls_names)]
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        frame[:] = 20
        for _ in range(60):
            bx, by = random.randint(0, size[0]), random.randint(0, size[1])
            cv2.circle(frame, (bx, by), random.randint(1, 3),
                       (random.randint(40, 70),) * 3, -1)
        cx, cy = random.randint(120, size[0] - 120), random.randint(100, size[1] - 100)
        r = random.randint(40, 60)
        DRAWERS[cls_name](frame, cx, cy, r, random.choice(COLORS))
        cv2.putText(frame, cls_name.replace("_", " "), (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 180, 216), 2)
        writer.write(frame)
    writer.release()
    print(f"Generated video at {path} ({duration}s @ {fps}fps)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--per-class", type=int, default=12)
    parser.add_argument("--video", default="test_micro.mp4")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    generate_dataset(args.dataset, args.per_class)
    if not args.no_video:
        generate_video(args.video, list(DRAWERS.keys()))


if __name__ == "__main__":
    main()
