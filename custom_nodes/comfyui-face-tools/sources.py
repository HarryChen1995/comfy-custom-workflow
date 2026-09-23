"""Multiple SOURCE PHOTOS, numbered, instead of one photo holding many faces.

SourcePhotoSet  - chain one node per photo; each gets an index and a label
FaceSwapMulti   - swap by index mapping "photo > targetFace", any batch size
"""
import os

import numpy as np
from PIL import Image, ImageDraw

from .identity import (_reactor, _to_np, _to_tensor_batch, _embed, _analyze, _font)

ORDERS = ["large-small", "small-large", "left-right", "right-left",
          "top-bottom", "bottom-top"]
TILE_COLORS = [(0, 220, 90), (255, 190, 0), (0, 170, 255), (255, 80, 200),
               (255, 90, 60), (140, 230, 255), (200, 120, 255), (120, 255, 170)]


def _pick_face(rs, rgb, order, det_size, face_index):
    import cv2
    faces = _analyze(rs, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), det_size)
    faces = rs.sort_by_order(faces, order) if faces else []
    if not faces:
        return None, 0
    idx = min(max(0, int(face_index)), len(faces) - 1)
    return faces[idx], len(faces)


def _sheet(photos, tile=150, cols=6):
    """Numbered contact sheet - one tile per PHOTO, not per face."""
    if not photos:
        img = Image.new("RGB", (tile * 3, tile), (28, 28, 32))
        ImageDraw.Draw(img).text((12, tile // 2 - 8), "no photos yet",
                                 font=_font(18), fill=(235, 235, 235))
        return img
    n = len(photos)
    cols = min(cols, n)
    rows = (n + cols - 1) // cols
    labh, gap = 26, 6
    W = cols * tile + (cols + 1) * gap
    H = rows * (tile + labh) + (rows + 1) * gap
    sheet = Image.new("RGB", (W, H), (28, 28, 32))
    d = ImageDraw.Draw(sheet)
    f = _font(15)
    for k, p in enumerate(photos):
        r, c = divmod(k, cols)
        ox = gap + c * (tile + gap)
        oy = gap + r * (tile + labh + gap)
        src = Image.fromarray(p["rgb"])
        rect = p.get("rect")
        if rect:
            x0, y0, x1, y1 = rect
            mx, my = int((x1 - x0) * 0.35), int((y1 - y0) * 0.35)
            src = src.crop((max(0, x0 - mx), max(0, y0 - my),
                            min(src.width, x1 + mx), min(src.height, y1 + my)))
        if src.width and src.height:
            sheet.paste(src.resize((tile, tile), Image.LANCZOS), (ox, oy))
        col = TILE_COLORS[k % len(TILE_COLORS)] if rect else (120, 120, 128)
        d.rectangle([ox, oy, ox + tile, oy + tile], outline=col, width=3)
        d.rectangle([ox, oy + tile, ox + tile, oy + tile + labh], fill=col)
        txt = "%d  %s" % (k, p["label"][:14])
        if not rect:
            txt = "%d  NO FACE" % k
        d.text((ox + 6, oy + tile + 4), txt, font=f, fill=(0, 0, 0))
    return sheet


class SourcePhotoSet:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "label": ("STRING", {"default": ""}),
                "face_index": ("INT", {"default": 0, "min": 0, "max": 100}),
                "order": (ORDERS, {"default": "large-small"}),
                "det_size": ("INT", {"default": 640, "min": 160, "max": 2048, "step": 32}),
            },
            "optional": {"append_to": ("SOURCE_SET",)},
        }

    RETURN_TYPES = ("SOURCE_SET", "IMAGE", "STRING")
    RETURN_NAMES = ("source_set", "sheet", "report")
    FUNCTION = "run"
    CATEGORY = "face tools"

    def run(self, image, label, face_index, order, det_size, append_to=None):
        rs = _reactor()
        prior = list(append_to.get("photos", [])) if isinstance(append_to, dict) else []
        rgb = _to_np(image)
        face, nfound = _pick_face(rs, rgb, order, det_size, face_index)
        entry = {"rgb": rgb, "label": (label.strip() or "photo %d" % len(prior)),
                 "face_index": int(face_index), "order": order,
                 "det_size": int(det_size), "rect": None}
        if face is not None:
            x0, y0, x1, y1 = [int(v) for v in face.bbox]
            entry["rect"] = (max(0, x0), max(0, y0),
                             min(rgb.shape[1], x1), min(rgb.shape[0], y1))
        photos = prior + [entry]
        lines = []
        for k, p in enumerate(photos):
            lines.append("[%d] %-16s %s" % (k, p["label"],
                                            "ok" if p["rect"] else "NO FACE DETECTED"))
        rep = "%d photo(s)\n%s" % (len(photos), "\n".join(lines))
        if face is None:
            rep += "\nWARNING: this photo has no usable face (%d found)" % nfound
        print("[SourcePhotoSet] " + rep)
        return ({"photos": photos}, _to_tensor_batch([np.array(_sheet(photos))]), rep)


def _parse_pairs(spec):
    out = []
    for chunk in spec.replace(";", "\n").replace(",", "\n").split("\n"):
        c = chunk.strip()
        if not c:
            continue
        sep = ">" if ">" in c else (":" if ":" in c else None)
        if sep is None:
            continue
        a, _, b = c.partition(sep)
        try:
            out.append((int(a.strip()), int(b.strip())))
        except ValueError:
            continue
    return out


class FaceSwapMulti:
    """Index mapping 'sourcePhoto > targetFace', with one photo per person."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "source_set": ("SOURCE_SET",),
                "mapping": ("STRING", {"default": "0>0", "multiline": True}),
                "order": (ORDERS, {"default": "large-small"}),
                "det_size": ("INT", {"default": 640, "min": 160, "max": 2048, "step": 32}),
                "swap_model": ("STRING", {"default": "inswapper_128.onnx"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "report")
    FUNCTION = "run"
    CATEGORY = "face tools"

    def run(self, images, source_set, mapping, order, det_size, swap_model):
        import cv2
        import folder_paths
        rs = _reactor()

        photos = source_set.get("photos", []) if isinstance(source_set, dict) else []
        if not photos:
            raise RuntimeError("source_set is empty - chain at least one Source Photo Set")
        pairs = _parse_pairs(mapping)
        if not pairs:
            raise RuntimeError("mapping is empty - use lines like  0>2")

        warn = []
        srcfaces = {}
        for pi, _ in pairs:
            if pi in srcfaces:
                continue
            if not (0 <= pi < len(photos)):
                warn.append("photo %d does not exist (%d loaded)" % (pi, len(photos)))
                continue
            p = photos[pi]
            f, _n = _pick_face(rs, p["rgb"], p["order"], p["det_size"], p["face_index"])
            if f is None:
                warn.append("photo %d (%s) has no usable face" % (pi, p["label"]))
            else:
                srcfaces[pi] = f

        model_path = folder_paths.get_full_path("insightface", swap_model)
        if model_path is None:
            model_path = os.path.join(folder_paths.models_dir, "insightface", swap_model)
        swapper = rs.getFaceSwapModel(model_path)

        n = images.shape[0]
        out, done = [], 0
        for fi in range(n):
            rgb = _to_np(images, fi)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            faces = _analyze(rs, bgr, det_size)
            faces = rs.sort_by_order(faces, order) if faces else []
            touched = False
            used = set()
            for pi, ti in pairs:
                if pi not in srcfaces or ti in used or ti >= len(faces):
                    continue
                bgr = swapper.get(bgr, faces[ti], srcfaces[pi], paste_back=True)
                used.add(ti)
                touched = True
            out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if touched else rgb)
            if touched:
                done += 1
            if n > 20 and ((fi + 1) % 25 == 0 or fi == n - 1):
                print("[FaceSwapMulti] %d/%d frames, %d touched" % (fi + 1, n, done),
                      flush=True)

        lines = ["%d image(s) | %d changed" % (n, done)]
        for pi, ti in pairs:
            nm = photos[pi]["label"] if 0 <= pi < len(photos) else "?"
            lines.append("  photo %d (%s)  ->  target face %d" % (pi, nm, ti))
        if warn:
            lines.append("WARNING:")
            lines += ["  " + w for w in warn]
        rep = "\n".join(lines)
        print("[FaceSwapMulti] " + rep)
        return (_to_tensor_batch(out), rep)


NODE_CLASS_MAPPINGS = {
    "SourcePhotoSet": SourcePhotoSet,
    "FaceSwapMulti": FaceSwapMulti,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SourcePhotoSet": "Source Photo Set (chain one per photo)",
    "FaceSwapMulti": "Face Swap Multi (photo -> target face)",
}
