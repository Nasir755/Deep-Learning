"""
separate_receipts.py
--------------------
Ek photo mai maujood sab receipts ko dhoond kar:
  1) Har receipt ke gird HARA (green) border + label ("Slip 1: 354x644") lagata hai
  2) Har receipt ko alag alag image mai crop karke save karta hai
  3) Har receipt ka saaf (white background, black text) version bhi save karta hai

Do tareeqe (mode) khud hi chun leta hai:
  - Agar image mai NEELI (blue) lines lagi hain  -> un lines ke andar wala hissa receipt maanta hai
  - Warna (sadi photo)                            -> text ki position se receipts khud dhoondta hai

Use:
    python separate_receipts.py input_image.jpeg
    python separate_receipts.py input_image.jpeg --out-dir result

Requirements:
    pip install opencv-python numpy
"""

import argparse
import os

import cv2
import numpy as np


# ----------------------------------------------------------------------------
# Unicode-safe image read / write (Windows path ke masle se bachne ke liye)
# ----------------------------------------------------------------------------
def imread(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def imwrite(path, img):
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f"Save nahi hui: {path}")
    buf.tofile(path)


# ----------------------------------------------------------------------------
# MODE 1: blue lines wali image
# ----------------------------------------------------------------------------
def blue_line_boxes(img):
    """Blue lines ke andar band hone wale regions. Return: (boxes, polygons) ya ([], [])."""
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, np.array([98, 90, 60]), np.array([135, 255, 255]))
    if (blue > 0).mean() < 0.003:  # neeli lines hain hi nahi
        return [], []

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, k, iterations=2)
    blue = cv2.dilate(blue, k)
    # Jo receipt image ke kinare se kat rahi ho, uske liye border ko bhi line maan lo
    b = 4
    blue[:b, :] = blue[-b:, :] = 255
    blue[:, :b] = blue[:, -b:] = 255

    contours, hier = cv2.findContours(blue, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return [], []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    boxes, polys = [], []
    for cnt, hh in zip(contours, hier[0]):
        if hh[3] == -1 or cv2.contourArea(cnt) < h * w * 0.02:
            continue
        m = np.zeros((h, w), np.uint8)
        cv2.drawContours(m, [cnt], -1, 255, -1)
        if (gray[m > 0] < 110).mean() < 0.02:  # text nahi => khali background
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        boxes.append((x, y, x + bw, y + bh))
        polys.append(cnt)
    return boxes, polys


# ----------------------------------------------------------------------------
# MODE 2: sadi photo -> text ki position se receipts dhoondna (XY-cut)
# ----------------------------------------------------------------------------
def remove_colored_lines(img):
    """Koi rangeen (neeli/hari/cyan) lines ho to unhe inpaint karke hata deta hai."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 1] > 110) & (hsv[..., 2] > 70)).astype(np.uint8) * 255
    if (m > 0).mean() < 0.0005:
        return img
    m = cv2.dilate(m, np.ones((7, 7), np.uint8))
    return cv2.inpaint(img, m, 3, cv2.INPAINT_TELEA)


def ink_mask(img):
    """Kaale (text) pixels ka mask."""
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 41, 18)
    n, lab, st, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= 8
    return (keep[lab] * 255).astype(np.uint8)


def drop_specks(ink, min_ink=250, k=41):
    """Kagaz ke kinare ki lakeerein / dhabbe jaise faltu nishan hata deta hai."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    bw, bh = st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT]
    thin = (np.minimum(bw, bh) <= 4) & (np.maximum(bw, bh) <= 60)
    thin[0] = True
    ink = (ink * (~thin)[lab]).astype(np.uint8)

    d = cv2.dilate(ink, np.ones((k, k), np.uint8))
    n, lab, _, _ = cv2.connectedComponentsWithStats(d, connectivity=8)
    cnt = np.bincount(lab.ravel(), weights=(ink.ravel() > 0).astype(float), minlength=n)
    keep = cnt >= min_ink
    keep[0] = False
    return (ink * keep[lab]).astype(np.uint8)


def empty_runs(profile, noise):
    empty = profile <= noise
    runs, i, n = [], 0, len(empty)
    while i < n:
        if empty[i]:
            j = i
            while j < n and empty[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def xy_cut(mask, x0, y0, x1, y1, gap_x, gap_y, nf=0.02):
    """Recursive XY-cut: sabse bade khali gap par region ko do hisson mai baant deta hai."""
    sub = mask[y0:y1, x0:x1]
    ys, xs = np.where(sub > 0)
    if len(xs) == 0:
        return []
    x0, x1 = x0 + int(xs.min()), x0 + int(xs.max()) + 1
    y0, y1 = y0 + int(ys.min()), y0 + int(ys.max()) + 1
    sub = mask[y0:y1, x0:x1]
    H, W = sub.shape
    cols, rows = (sub > 0).sum(0), (sub > 0).sum(1)

    best = None
    for axis, prof, g, noise in (("x", cols, gap_x, max(2, nf * H)),
                                 ("y", rows, gap_y, max(2, nf * W))):
        for a, b in empty_runs(prof, noise):
            if a == 0 or b == len(prof):
                continue
            if b - a >= g:
                score = (b - a) / g
                if best is None or score > best[0]:
                    best = (score, axis, a, b)
    if best is None:
        return [(x0, y0, x1, y1)]
    _, axis, a, b = best
    if axis == "x":
        return (xy_cut(mask, x0, y0, x0 + a, y1, gap_x, gap_y, nf) +
                xy_cut(mask, x0 + b, y0, x1, y1, gap_x, gap_y, nf))
    return (xy_cut(mask, x0, y0, x1, y0 + a, gap_x, gap_y, nf) +
            xy_cut(mask, x0, y0 + b, x1, y1, gap_x, gap_y, nf))


def box_distance(a, b):
    dx = max(a[0] - b[2], b[0] - a[2], 0)
    dy = max(a[1] - b[3], b[1] - a[3], 0)
    return max(dx, dy)


def auto_boxes(img):
    h, w = img.shape[:2]
    clean = remove_colored_lines(img)
    ink = drop_specks(ink_mask(clean))
    frags = xy_cut(ink, 0, 0, w, h, int(w * 0.025), int(h * 0.04))

    def ink_count(b):
        return int((ink[b[1]:b[3], b[0]:b[2]] > 0).sum())

    big = [b for b in frags if (b[2] - b[0]) * (b[3] - b[1]) > 0.03 * w * h and ink_count(b) > 1500]
    small = [b for b in frags if b not in big]

    # Chhote tukre (jaise receipt ka sirf title) ko sabse qareebi receipt mai jod do
    for s in small:
        if not big or ink_count(s) < 300:
            continue
        j = min(range(len(big)), key=lambda i: box_distance(s, big[i]))
        if box_distance(s, big[j]) < 0.08 * h:
            b = big[j]
            big[j] = (min(b[0], s[0]), min(b[1], s[1]), max(b[2], s[2]), max(b[3], s[3]))
    return big


def pad_boxes(boxes, w, h, pad):
    """Har box ko thora bada karo, lekin aapas mai overlap na ho."""
    out = [[max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)]
           for x0, y0, x1, y1 in boxes]
    for i in range(len(out)):
        for j in range(i + 1, len(out)):
            a, b = out[i], out[j]
            ox = min(a[2], b[2]) - max(a[0], b[0])
            oy = min(a[3], b[3]) - max(a[1], b[1])
            if ox <= 0 or oy <= 0:
                continue
            if ox < oy:  # x direction mai chhota overlap -> x mai alag karo
                mid = (max(a[0], b[0]) + min(a[2], b[2])) // 2
                if a[0] < b[0]:
                    a[2], b[0] = mid, mid
                else:
                    b[2], a[0] = mid, mid
            else:
                mid = (max(a[1], b[1]) + min(a[3], b[3])) // 2
                if a[1] < b[1]:
                    a[3], b[1] = mid, mid
                else:
                    b[3], a[1] = mid, mid
    return [tuple(int(v) for v in b) for b in out]


# ----------------------------------------------------------------------------
# Common helpers
# ----------------------------------------------------------------------------
def reading_order(items, row_tol=0.5):
    """items = [(box, extra)], upar->neeche, left->right."""
    items = sorted(items, key=lambda t: t[0][1])
    rows, cur = [], []
    for it in items:
        if cur and it[0][1] >= cur[0][0][1] + (cur[0][0][3] - cur[0][0][1]) * row_tol:
            rows.append(cur)
            cur = []
        cur.append(it)
    if cur:
        rows.append(cur)
    return [it for r in rows for it in sorted(r, key=lambda t: t[0][0])]


def clean_receipt(img_bgr, upscale=2):
    """White background + gehra black text."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if upscale != 1:
        gray = cv2.resize(gray, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    k = max(15, (min(gray.shape) // 25) | 1)
    bg = cv2.dilate(gray, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(cv2.medianBlur(bg, k), (0, 0), k / 2)
    norm = cv2.divide(gray, bg, scale=255)
    lo, hi = 85, 205
    out = np.clip((norm.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    e = 2 * upscale
    out[:e, :] = out[-e:, :] = 255
    out[:, :e] = out[:, -e:] = 255
    return out


def draw_annotations(img, boxes):
    vis = img.copy()
    h, w = vis.shape[:2]
    thick = max(2, w // 350)
    scale = max(0.5, w / 1600)
    for i, (x0, y0, x1, y1) in enumerate(boxes, 1):
        green = (0, 220, 0)
        cv2.rectangle(vis, (x0, y0), (x1 - 1, y1 - 1), green, thick)
        label = f"Slip {i}: {x1 - x0}x{y1 - y0}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        ty = y0 - 6 if y0 - th - 10 > 0 else y0 + th + 8
        cv2.rectangle(vis, (x0, ty - th - 4), (x0 + tw + 8, ty + base), green, -1)
        cv2.putText(vis, label, (x0 + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick,
                    cv2.LINE_AA)
    return vis


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="input image")
    ap.add_argument("--out-dir", default="result", help="output folder")
    ap.add_argument("--auto", action="store_true",
                    help="blue lines ko ignore karke receipts khud dhoondo")
    args = ap.parse_args()

    img = imread(args.input)
    if img is None:
        raise SystemExit(f"Image nahi khuli: {args.input}")
    h, w = img.shape[:2]

    polys = None
    boxes = []
    if not args.auto:
        boxes, polys = blue_line_boxes(img)
    if len(boxes) >= 2:
        print(f"Mode: blue lines  ({len(boxes)} receipts)")
        items = reading_order(list(zip(boxes, polys)))
        boxes = [b for b, _ in items]
        polys = [p for _, p in items]
    else:
        polys = None
        boxes = auto_boxes(img)
        print(f"Mode: auto detect  ({len(boxes)} receipts)")
        boxes = [b for b, _ in reading_order([(b, None) for b in boxes])]
        boxes = pad_boxes(boxes, w, h, pad=max(6, int(w * 0.012)))

    if not boxes:
        raise SystemExit("Koi receipt nahi mili.")

    os.makedirs(os.path.join(args.out_dir, "slips"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "slips_clean"), exist_ok=True)

    source = remove_colored_lines(img)  # crops mai rangeen lines na aayein
    for i, (x0, y0, x1, y1) in enumerate(boxes, 1):
        crop = source[y0:y1, x0:x1].copy()
        if polys is not None:  # blue mode: receipt ke bahar wala hissa white kar do
            m = np.zeros((h, w), np.uint8)
            cv2.drawContours(m, [polys[i - 1]], -1, 255, -1)
            m = cv2.erode(m, np.ones((9, 9), np.uint8))[y0:y1, x0:x1]
            crop[m == 0] = (255, 255, 255)
        imwrite(os.path.join(args.out_dir, "slips", f"slip_{i}.png"), crop)
        imwrite(os.path.join(args.out_dir, "slips_clean", f"slip_{i}.png"), clean_receipt(crop))
        print(f"Slip {i}: box=({x0},{y0},{x1},{y1})  size={x1 - x0}x{y1 - y0}")

    ann = os.path.join(args.out_dir, "annotated.jpg")
    imwrite(ann, draw_annotations(source, boxes))
    print("Annotated image:", ann)
    print("Alag receipts:", os.path.join(args.out_dir, "slips"))


if __name__ == "__main__":
    main()