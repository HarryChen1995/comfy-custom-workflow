"""Face detection preview + manual source->target mapping for ReActor.

FaceDetectPreview - renders numbered boxes so you can SEE each face index
FaceFaceMapping   - turn "0>2, 1>5" into the two index strings ReActor wants

Indices come from ReActor's OWN analyze_faces + sort_by_order, so the numbers
drawn here are exactly the numbers ReActor will use when swapping.
"""
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REACTOR_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "ComfyUI-ReActor")

ORDERS = ["large-small", "small-large", "left-right", "right-left",
          "top-bottom", "bottom-top"]

# palette cycles so adjacent faces never share a colour
COLORS = [(0, 220, 90), (255, 190, 0), (0, 170, 255), (255, 80, 200),
          (255, 90, 60), (140, 230, 255), (200, 120, 255), (120, 255, 170),
          (255, 235, 120), (90, 140, 255)]


def _reactor():
    """Import ReActor's own detector so our indices match its swap order."""
    if REACTOR_DIR not in sys.path:
        sys.path.insert(0, REACTOR_DIR)
    from scripts.reactor_swapper import analyze_faces, sort_by_order
    return analyze_faces, sort_by_order


def _allow_det_size(px):
    """ReActor seeds ANALYSIS_MODELS with only '320'/'640' and indexes it with
    a bare [] lookup, so any other det_size raises KeyError. Seed the key so
    arbitrary detector sizes work, without patching ReActor itself."""
    try:
        if REACTOR_DIR not in sys.path:
            sys.path.insert(0, REACTOR_DIR)
        from scripts import reactor_swapper as rs
        rs.ANALYSIS_MODELS.setdefault(str(int(px)), None)
    except Exception as e:
        print("[face-tools] could not pre-seed det_size %s: %s" % (px, e))


def _auto_det_size(w, h):
    """Bigger detector input = small faces land on more pixels. Scale with the
    image but stay at 640 for ordinary photos and cap so VRAM stays sane."""
    px = int(round(max(w, h) / 2.0 / 32.0) * 32)
    return max(640, min(1280, px))


def _to_np(image):
    arr = image[0].detach().cpu().numpy()
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def _to_tensor(arr):
    return torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(0)


def _font(size):
    for n in ("arialbd.ttf", "arial.ttf", "segoeuib.ttf"):
        p = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", n)
        if os.path.isfile(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _parse_pairs(spec):
    """'0>2, 1>5' or newline separated -> [(0,2),(1,5)]. Also accepts '0:2'."""
    out = []
    for chunk in spec.replace(";", "\n").replace(",", "\n").split("\n"):
        c = chunk.strip()
        if not c:
            continue
        sep = ">" if ">" in c else (":" if ":" in c else ("-" if "-" in c[1:] else None))
        if sep is None:
            continue
        a, _, b = c.partition(sep)
        try:
            out.append((int(a.strip()), int(b.strip())))
        except ValueError:
            continue
    return out


class FaceDetectPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "order": (ORDERS, {"default": "large-small"}),
                "det_size": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 32}),
                "label_scale": ("FLOAT", {"default": 1.0, "min": 0.3, "max": 4.0, "step": 0.1}),
                "show_details": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "FACE_BOXES", "STRING", "INT")
    RETURN_NAMES = ("preview", "face_grid", "faces", "report", "count")
    FUNCTION = "run"
    CATEGORY = "face tools"

    def run(self, image, order, det_size, label_scale, show_details):
        import cv2
        analyze_faces, sort_by_order = _reactor()

        rgb = _to_np(image)
        h, w = rgb.shape[:2]
        ds = _auto_det_size(w, h) if int(det_size) <= 0 else int(det_size)
        ds = max(160, int(round(ds / 32.0) * 32))    # detector wants /32
        _allow_det_size(ds)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)   # ReActor analyses BGR
        faces = analyze_faces(bgr, det_size=(ds, ds))
        faces = sort_by_order(faces, order) if faces else []

        prev = Image.fromarray(rgb.copy())
        d = ImageDraw.Draw(prev)

        boxes, lines = [], []
        for idx, fc in enumerate(faces):
            x0, y0, x1, y1 = [int(v) for v in fc.bbox]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            bw, bh = x1 - x0, y1 - y0
            col = COLORS[idx % len(COLORS)]

            lw = max(1, int(round(bh * 0.05)))
            d.rectangle([x0, y0, x1, y1], outline=col, width=lw)

            # label scales with THIS face, so it never swamps a small one
            fs = int(max(9, min(bh * 0.55, 64)) * float(label_scale))
            f = _font(fs)
            tag = str(idx)
            tb = d.textbbox((0, 0), tag, font=f)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            pad = max(1, fs // 6)
            lw_box, lh_box = tw + pad * 2, th + pad * 2
            # sit the tag above the box when there is room, else just inside
            ly = y0 - lh_box if y0 - lh_box >= 0 else y0
            lx = min(x0, w - lw_box)
            d.rectangle([lx, ly, lx + lw_box, ly + lh_box], fill=col)
            d.text((lx + pad - tb[0], ly + pad - tb[1]), tag, font=f, fill=(0, 0, 0))

            g = getattr(fc, "gender", None)
            age = getattr(fc, "age", None)
            gs = "M" if g == 1 else ("F" if g == 0 else "?")
            boxes.append({"index": idx, "rect": (x0, y0, x1, y1),
                          "gender": gs, "age": age, "area": int(bw * bh)})
            lines.append("[%d] rect=(%d,%d,%d,%d) %dx%d %s%s"
                         % (idx, x0, y0, x1, y1, bw, bh, gs,
                            (" age~%s" % age) if age is not None else ""))

        grid = self._grid(rgb, boxes, show_details)
        report = ("%d face(s)  image=%dx%d  order=%s  det_size=%d%s\n"
                  % (len(boxes), w, h, order, ds,
                     " (auto)" if int(det_size) <= 0 else "")) + \
                 ("\n".join(lines) if lines else "none detected")
        print("[FaceDetectPreview] " + report)
        return (_to_tensor(np.array(prev)), _to_tensor(np.array(grid)),
                {"faces": boxes, "size": (w, h), "order": order},
                report, len(boxes))

    def _grid(self, rgb, boxes, show_details, tile=128, cols=6):
        """Contact sheet of numbered face crops - readable when faces are tiny."""
        if not boxes:
            img = Image.new("RGB", (tile * 3, tile), (28, 28, 32))
            d = ImageDraw.Draw(img)
            d.text((12, tile // 2 - 10), "no faces detected", font=_font(20),
                   fill=(235, 235, 235))
            return img
        src = Image.fromarray(rgb)
        n = len(boxes)
        cols = min(cols, n)
        rows = (n + cols - 1) // cols
        labh = 26
        gap = 6
        W = cols * tile + (cols + 1) * gap
        H = rows * (tile + labh) + (rows + 1) * gap
        sheet = Image.new("RGB", (W, H), (28, 28, 32))
        d = ImageDraw.Draw(sheet)
        f = _font(16)
        for k, b in enumerate(boxes):
            r, c = divmod(k, cols)
            ox = gap + c * (tile + gap)
            oy = gap + r * (tile + labh + gap)
            x0, y0, x1, y1 = b["rect"]
            mx, my = int((x1 - x0) * 0.25), int((y1 - y0) * 0.25)
            crop = src.crop((max(0, x0 - mx), max(0, y0 - my),
                             min(src.width, x1 + mx), min(src.height, y1 + my)))
            if crop.width and crop.height:
                crop = crop.resize((tile, tile), Image.LANCZOS)
                sheet.paste(crop, (ox, oy))
            col = COLORS[k % len(COLORS)]
            d.rectangle([ox, oy, ox + tile, oy + tile], outline=col, width=3)
            txt = str(b["index"])
            if show_details:
                txt += "   %s%s" % (b["gender"],
                                    (" %s" % b["age"]) if b["age"] is not None else "")
            d.rectangle([ox, oy + tile, ox + tile, oy + tile + labh], fill=col)
            d.text((ox + 6, oy + tile + 4), txt, font=f, fill=(0, 0, 0))
        return sheet


class FaceFaceMapping:
    """Manual mapping: 'sourceFace > targetFace', one pair per line."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mapping": ("STRING", {"default": "0>0", "multiline": True}),
            },
            "optional": {
                "source_faces": ("FACE_BOXES",),
                "target_faces": ("FACE_BOXES",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("source_faces_index", "input_faces_index", "report")
    FUNCTION = "run"
    CATEGORY = "face tools"

    def run(self, mapping, source_faces=None, target_faces=None):
        pairs = _parse_pairs(mapping)
        ns = len(source_faces.get("faces", [])) if isinstance(source_faces, dict) else None
        nt = len(target_faces.get("faces", [])) if isinstance(target_faces, dict) else None

        good, warn = [], []
        for s, t in pairs:
            if ns is not None and not (0 <= s < ns):
                warn.append("source face %d does not exist (source has %d)" % (s, ns))
                continue
            if nt is not None and not (0 <= t < nt):
                warn.append("target face %d does not exist (target has %d)" % (t, nt))
                continue
            good.append((s, t))

        seen, dedup = set(), []
        for s, t in good:
            if t in seen:
                warn.append("target face %d mapped more than once - keeping the first" % t)
                continue
            seen.add(t)
            dedup.append((s, t))

        src = ",".join(str(s) for s, _ in dedup)
        tgt = ",".join(str(t) for _, t in dedup)
        lines = ["source %d  ->  target %d" % (s, t) for s, t in dedup]
        report = ("%d mapping(s)\n" % len(dedup)) + "\n".join(lines)
        if ns is not None:
            report += "\nsource image has %d face(s)" % ns
        if nt is not None:
            report += "\ntarget image has %d face(s)" % nt
        if warn:
            report += "\nWARNING:\n  " + "\n  ".join(warn)
        if not dedup:
            report += "\nNothing mapped - swap will do nothing."
        print("[FaceFaceMapping] source_faces_index=%r input_faces_index=%r\n%s"
              % (src, tgt, report))
        return (src, tgt, report)


NODE_CLASS_MAPPINGS = {
    "FaceDetectPreview": FaceDetectPreview,
    "FaceFaceMapping": FaceFaceMapping,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceDetectPreview": "Face Detect Preview (numbered)",
    "FaceFaceMapping": "Face Mapping (source -> target)",
}

# identity-matched swap lives in its own module
try:
    from .identity import (NODE_CLASS_MAPPINGS as _ID_C,
                           NODE_DISPLAY_NAME_MAPPINGS as _ID_D)
    NODE_CLASS_MAPPINGS.update(_ID_C)
    NODE_DISPLAY_NAME_MAPPINGS.update(_ID_D)
except Exception as _e:
    print("[face-tools] identity nodes unavailable: %s" % _e)

# multi-person identity mapping
try:
    from .multi import (NODE_CLASS_MAPPINGS as _M_C,
                        NODE_DISPLAY_NAME_MAPPINGS as _M_D)
    NODE_CLASS_MAPPINGS.update(_M_C)
    NODE_DISPLAY_NAME_MAPPINGS.update(_M_D)
except Exception as _e:
    print("[face-tools] multi-person nodes unavailable: %s" % _e)

# multiple source PHOTOS instead of one photo with many faces
try:
    from .sources import (NODE_CLASS_MAPPINGS as _S_C,
                          NODE_DISPLAY_NAME_MAPPINGS as _S_D)
    NODE_CLASS_MAPPINGS.update(_S_C)
    NODE_DISPLAY_NAME_MAPPINGS.update(_S_D)
except Exception as _e:
    print("[face-tools] source-set nodes unavailable: %s" % _e)
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
