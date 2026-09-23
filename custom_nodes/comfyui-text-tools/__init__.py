"""Automatic text detection / erase / re-draw helpers for ComfyUI.

TextDetectMask   - OCR the image, emit a MASK over the chosen text regions
DrawTextInBoxes  - render replacement text back into those regions
"""
import os
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageColor

_READER = {}


def _get_reader(langs, gpu):
    key = (langs, gpu)
    if key not in _READER:
        import easyocr
        # pin to one GPU - EasyOCR otherwise DataParallels across both cards
        _READER[key] = easyocr.Reader(
            [s.strip() for s in langs.split(",") if s.strip()],
            gpu=gpu, verbose=False,
        )
    return _READER[key]


def _font_dirs():
    return [
        os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"),
        os.path.join(os.path.dirname(__file__), "fonts"),
    ]


def _list_fonts():
    out, seen = [], set()
    for d in _font_dirs():
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".ttf", ".otf")) and f.lower() not in seen:
                seen.add(f.lower())
                out.append(f)
    return out or ["arial.ttf"]


def _font_path(name):
    for d in _font_dirs():
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return None


def _to_np(image):
    """ComfyUI IMAGE [B,H,W,C] float 0-1 -> uint8 HWC RGB of first frame."""
    arr = image[0].detach().cpu().numpy()
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def _to_tensor(arr):
    return torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(0)


def _quad_to_rect(box, pad, w, h):
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    x0 = max(0, int(min(xs)) - pad)
    y0 = max(0, int(min(ys)) - pad)
    x1 = min(w, int(max(xs)) + pad)
    y1 = min(h, int(max(ys)) + pad)
    return x0, y0, x1, y1


def _parse_indices(spec, n):
    """'0,2-4' -> {0,2,3,4}. Empty string means 'everything'."""
    spec = (spec or "").strip()
    if not spec:
        return set(range(n))
    picked = set()
    for part in spec.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part[1:]:
            a, _, b = part.partition("-")
            try:
                lo, hi = int(a), int(b)
            except ValueError:
                continue
            for i in range(min(lo, hi), max(lo, hi) + 1):
                if 0 <= i < n:
                    picked.add(i)
        else:
            try:
                i = int(part)
            except ValueError:
                continue
            if 0 <= i < n:
                picked.add(i)
    return picked


def _parse_color(spec, fallback=(0, 0, 0)):
    """'auto' handled by caller. Accepts #rrggbb, named colours, rgb(r,g,b)."""
    s = (spec or "").strip()
    if not s:
        return fallback
    try:
        c = ImageColor.getrgb(s)
        return tuple(int(v) for v in c[:3])
    except ValueError:
        pass
    h = s.lstrip("#")
    if len(h) == 6:
        try:
            return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
        except ValueError:
            pass
    return fallback


def _glyph_mask(crop_rgb, dilate):
    """Mask only the letter pixels inside a crop, not the whole rectangle.

    Far kinder to LaMa: it fills thin strokes convincingly but smears when
    asked to invent a large rectangle of structured background.
    """
    import cv2
    if crop_rgb.size == 0:
        return np.zeros(crop_rgb.shape[:2], dtype=np.float32)
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = (th > 0)
    # the glyphs are whichever cluster covers less of the crop
    if fg.sum() > fg.size / 2:
        fg = ~fg
    m = fg.astype(np.uint8) * 255
    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate * 2 + 1, dilate * 2 + 1))
        m = cv2.dilate(m, k)
    return (m.astype(np.float32) / 255.0)


class TextDetectMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "languages": ("STRING", {"default": "en"}),
                "confidence": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.05}),
                "detect_scale": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 4.0, "step": 0.25}),
                "padding": ("INT", {"default": 6, "min": 0, "max": 200, "step": 1}),
                "select": ("STRING", {"default": ""}),
                "contains": ("STRING", {"default": ""}),
                "invert_selection": ("BOOLEAN", {"default": False}),
                "mask_mode": (["box", "glyphs"], {"default": "box"}),
                "glyph_dilate": ("INT", {"default": 2, "min": 0, "max": 30, "step": 1}),
                "use_gpu": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MASK", "TEXT_BOXES", "STRING", "IMAGE")
    RETURN_NAMES = ("mask", "boxes", "detected_text", "preview")
    FUNCTION = "detect"
    CATEGORY = "text tools"

    def detect(self, image, languages, confidence, detect_scale, padding, select,
               contains, invert_selection, mask_mode, glyph_dilate, use_gpu):
        rgb = _to_np(image)
        h, w = rgb.shape[:2]
        reader = _get_reader(languages, bool(use_gpu))

        # small / low-contrast text scores far better when OCR sees it bigger;
        # detect on an upscaled copy, then map the boxes back to native coords
        s = float(detect_scale)
        if abs(s - 1.0) > 1e-3:
            big = Image.fromarray(rgb).resize(
                (max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
            results = reader.readtext(np.array(big))
            results = [([[p[0] / s, p[1] / s] for p in box], txt, conf)
                       for box, txt, conf in results]
        else:
            results = reader.readtext(rgb)

        # stable, human-predictable order: top-to-bottom then left-to-right
        found = []
        for box, txt, conf in results:
            if conf < confidence:
                continue
            found.append((_quad_to_rect(box, padding, w, h), txt, float(conf)))
        found.sort(key=lambda r: (r[0][1] // max(1, h // 40), r[0][0]))

        needle = contains.strip().lower()
        idx = _parse_indices(select, len(found))
        if needle:
            idx = {i for i in idx if needle in found[i][1].lower()}
        if invert_selection:
            idx = set(range(len(found))) - idx

        mask = np.zeros((h, w), dtype=np.float32)
        boxes, lines = [], []
        prev = Image.fromarray(rgb.copy())
        pd = ImageDraw.Draw(prev)
        try:
            pf = ImageFont.truetype(_font_path("arialbd.ttf") or "arial.ttf", 22)
        except Exception:
            pf = ImageFont.load_default()

        for i, (rect, txt, conf) in enumerate(found):
            x0, y0, x1, y1 = rect
            chosen = i in idx
            if chosen and x1 > x0 and y1 > y0:
                if mask_mode == "glyphs":
                    mask[y0:y1, x0:x1] = np.maximum(
                        mask[y0:y1, x0:x1],
                        _glyph_mask(rgb[y0:y1, x0:x1], int(glyph_dilate)))
                else:
                    mask[y0:y1, x0:x1] = 1.0
                boxes.append({"rect": (x0, y0, x1, y1), "text": txt, "conf": conf, "index": i})
            col = (0, 255, 0) if chosen else (255, 60, 60)
            pd.rectangle([x0, y0, x1, y1], outline=col, width=3)
            tag = str(i)
            tb = pd.textbbox((0, 0), tag, font=pf)
            pd.rectangle([x0, max(0, y0 - (tb[3] - tb[1]) - 8), x0 + (tb[2] - tb[0]) + 10, y0],
                         fill=col)
            pd.text((x0 + 5, max(0, y0 - (tb[3] - tb[1]) - 6)), tag, font=pf, fill=(0, 0, 0))
            lines.append("[%d]%s %r conf=%.2f rect=(%d,%d,%d,%d)"
                         % (i, " *" if chosen else "  ", txt, conf, x0, y0, x1, y1))

        report = "\n".join(lines) if lines else "no text detected"
        print("[TextDetectMask] %d detected, %d selected (* = selected)\n%s"
              % (len(found), len(boxes), report))
        return (torch.from_numpy(mask).unsqueeze(0),
                {"boxes": boxes, "size": (w, h)},
                report,
                _to_tensor(np.array(prev)))


class DrawTextInBoxes:
    @classmethod
    def INPUT_TYPES(cls):
        fonts = _list_fonts()
        default = "arialbd.ttf" if _font_path("arialbd.ttf") else fonts[0]
        return {
            "required": {
                "image": ("IMAGE",),
                "boxes": ("TEXT_BOXES",),
                "replacement_text": ("STRING", {"default": "NEW TEXT", "multiline": True}),
                "font": (fonts, {"default": default}),
                "color": ("STRING", {"default": "auto"}),
                "font_size": ("INT", {"default": 0, "min": 0, "max": 2000, "step": 1}),
                "bold": ("INT", {"default": 0, "min": 0, "max": 30, "step": 1}),
                "fill_ratio": ("FLOAT", {"default": 0.92, "min": 0.1, "max": 1.0, "step": 0.02}),
                "align": (["center", "left", "right"], {"default": "center"}),
                "valign": (["middle", "top", "bottom"], {"default": "middle"}),
                "offset_x": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1}),
                "offset_y": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1}),
                "rotation": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.5}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "outline_width": ("INT", {"default": 0, "min": 0, "max": 20, "step": 1}),
                "outline_color": ("STRING", {"default": "black"}),
            },
            "optional": {"color_reference": ("IMAGE",)},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "draw"
    CATEGORY = "text tools"

    def _pick_color(self, ref_rgb, rect):
        """Estimate original glyph colour via Otsu; minority cluster = text."""
        import cv2
        x0, y0, x1, y1 = rect
        crop = ref_rgb[y0:y1, x0:x1]
        if crop.size == 0:
            return (0, 0, 0)
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        fg = th > 0
        bg = th == 0
        if fg.sum() == 0 or bg.sum() == 0:
            return (0, 0, 0)
        sel = fg if fg.sum() <= bg.sum() else bg
        return tuple(int(v) for v in crop[sel].mean(axis=0))

    def _fit_font(self, text, path, box_w, box_h, ratio):
        if path is None:
            return ImageFont.load_default()
        lo, hi, best = 6, max(8, int(box_h * 2)), None
        while lo <= hi:
            mid = (lo + hi) // 2
            try:
                f = ImageFont.truetype(path, mid)
            except Exception:
                return ImageFont.load_default()
            l, t, r, b = f.getbbox(text)
            if (r - l) <= box_w * ratio and (b - t) <= box_h * ratio:
                best = f
                lo = mid + 1
            else:
                hi = mid - 1
        if best is not None:
            return best
        try:
            return ImageFont.truetype(path, 8)
        except Exception:
            return ImageFont.load_default()

    def draw(self, image, boxes, replacement_text, font, color, font_size, bold,
             fill_ratio, align, valign, offset_x, offset_y, rotation, opacity,
             outline_width, outline_color, color_reference=None):
        base = _to_np(image)
        ref = _to_np(color_reference) if color_reference is not None else base
        pil = Image.fromarray(base).convert("RGBA")
        # text goes on its own layer so it can be rotated / faded before compositing
        layer = Image.new("RGBA", pil.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        path = _font_path(font)

        blist = boxes.get("boxes", []) if isinstance(boxes, dict) else []
        texts = replacement_text.split("\n")
        # colour may be a single value or one per box, comma separated
        cspecs = [c.strip() for c in color.split(",")] if "," in color else [color.strip()]
        ocol = _parse_color(outline_color, (0, 0, 0))

        for i, b in enumerate(blist):
            if i < len(texts):
                new = texts[i]
            elif len(texts) == 1:
                new = texts[0]
            else:
                new = ""
            if not new.strip():
                continue
            x0, y0, x1, y1 = b["rect"]
            bw, bh = x1 - x0, y1 - y0

            cspec = cspecs[i] if i < len(cspecs) else cspecs[-1]
            if cspec.lower() == "auto":
                col = self._pick_color(ref, b["rect"])
            else:
                col = _parse_color(cspec, (0, 0, 0))

            if font_size > 0 and path is not None:
                try:
                    f = ImageFont.truetype(path, int(font_size))
                except Exception:
                    f = self._fit_font(new, path, bw, bh, fill_ratio)
            else:
                f = self._fit_font(new, path, bw, bh, fill_ratio)

            l, t, r, bm = f.getbbox(new)
            tw, th = r - l, bm - t
            if align == "left":
                tx = x0
            elif align == "right":
                tx = x1 - tw
            else:
                tx = x0 + (bw - tw) // 2
            if valign == "top":
                ty = y0 - t
            elif valign == "bottom":
                ty = y1 - th - t
            else:
                ty = y0 + (bh - th) // 2 - t
            pos = (tx - l + int(offset_x), ty + int(offset_y))
            bw_ = int(bold)
            # outline first (drawn wider), then the faux-bolded glyph on top
            if outline_width > 0:
                d.text(pos, new, font=f, fill=ocol,
                       stroke_width=bw_ + int(outline_width), stroke_fill=ocol)
            if bw_ > 0:
                d.text(pos, new, font=f, fill=col, stroke_width=bw_, stroke_fill=col)
            else:
                d.text(pos, new, font=f, fill=col)
            print("[DrawTextInBoxes] box %d (det #%s): %r colour=%s size=%s bold=%d"
                  % (i, b.get("index", "?"), new, col,
                     getattr(f, "size", "?"), bw_))
            if abs(float(rotation)) > 1e-3:
                # rotate this box's text about the box centre, then merge back
                cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                pad = max(bw, bh)
                lx0, ly0 = max(0, cx - bw - pad), max(0, cy - bh - pad)
                lx1, ly1 = min(layer.width, cx + bw + pad), min(layer.height, cy + bh + pad)
                patch = layer.crop((lx0, ly0, lx1, ly1))
                rot = patch.rotate(float(rotation), resample=Image.BICUBIC,
                                   center=(cx - lx0, cy - ly0))
                layer.paste(Image.new("RGBA", patch.size, (0, 0, 0, 0)), (lx0, ly0))
                layer.alpha_composite(rot, (lx0, ly0))

        if float(opacity) < 1.0:
            a = layer.getchannel("A").point(lambda v: int(v * float(opacity)))
            layer.putalpha(a)
        out = Image.alpha_composite(pil, layer).convert("RGB")
        return (_to_tensor(np.array(out)),)


def _boxes_to_mask(boxes, w, h):
    m = np.zeros((h, w), dtype=np.float32)
    for b in boxes:
        x0, y0, x1, y1 = b["rect"]
        m[y0:y1, x0:x1] = 1.0
    return torch.from_numpy(m).unsqueeze(0)


def _annotate(rgb, all_boxes, chosen_idx):
    prev = Image.fromarray(rgb.copy())
    pd = ImageDraw.Draw(prev)
    try:
        pf = ImageFont.truetype(_font_path("arialbd.ttf") or "arial.ttf", 22)
    except Exception:
        pf = ImageFont.load_default()
    for i, b in enumerate(all_boxes):
        x0, y0, x1, y1 = b["rect"]
        col = (0, 255, 0) if i in chosen_idx else (255, 60, 60)
        pd.rectangle([x0, y0, x1, y1], outline=col, width=3)
        tag = str(i)
        tb = pd.textbbox((0, 0), tag, font=pf)
        pd.rectangle([x0, max(0, y0 - (tb[3] - tb[1]) - 8),
                      x0 + (tb[2] - tb[0]) + 10, y0], fill=col)
        pd.text((x0 + 5, max(0, y0 - (tb[3] - tb[1]) - 6)), tag, font=pf, fill=(0, 0, 0))
    return _to_tensor(np.array(prev))


class SelectTextBoxes:
    """Narrow an existing TEXT_BOXES set - lets you erase one set but retype another."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "boxes": ("TEXT_BOXES",),
                "select": ("STRING", {"default": ""}),
                "contains": ("STRING", {"default": ""}),
                "invert_selection": ("BOOLEAN", {"default": False}),
            },
            "optional": {"image": ("IMAGE",)},
        }

    RETURN_TYPES = ("TEXT_BOXES", "MASK", "STRING", "IMAGE")
    RETURN_NAMES = ("boxes", "mask", "report", "preview")
    FUNCTION = "pick"
    CATEGORY = "text tools"

    def pick(self, boxes, select, contains, invert_selection, image=None):
        blist = boxes.get("boxes", []) if isinstance(boxes, dict) else []
        w, h = boxes.get("size", (0, 0)) if isinstance(boxes, dict) else (0, 0)
        idx = _parse_indices(select, len(blist))
        needle = contains.strip().lower()
        if needle:
            idx = {i for i in idx if needle in str(blist[i].get("text", "")).lower()}
        if invert_selection:
            idx = set(range(len(blist))) - idx

        out, lines = [], []
        for i, b in enumerate(blist):
            if i in idx:
                out.append(b)
            lines.append("[%d]%s %r rect=%s"
                         % (i, " *" if i in idx else "  ", b.get("text", ""), b["rect"]))
        report = "\n".join(lines) if lines else "no boxes on input"
        print("[SelectTextBoxes] %d in, %d selected\n%s" % (len(blist), len(out), report))

        if image is not None:
            rgb = _to_np(image)
            h, w = rgb.shape[:2]
            prev = _annotate(rgb, blist, idx)
        else:
            prev = torch.zeros((1, max(1, h), max(1, w), 3), dtype=torch.float32)
        return ({"boxes": out, "size": (w, h)}, _boxes_to_mask(out, w, h), report, prev)


class TextBoxesFromMask:
    """Paint a region in MaskEditor and turn it into an editable box."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "min_area": ("INT", {"default": 80, "min": 1, "max": 10000000, "step": 10}),
                "padding": ("INT", {"default": 0, "min": 0, "max": 200, "step": 1}),
            }
        }

    RETURN_TYPES = ("TEXT_BOXES", "STRING")
    RETURN_NAMES = ("boxes", "report")
    FUNCTION = "convert"
    CATEGORY = "text tools"

    def convert(self, mask, min_area, padding):
        import cv2
        m = mask[0].detach().cpu().numpy()
        h, w = m.shape[:2]
        binary = (m > 0.5).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        found = []
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if area < min_area:
                continue
            x0 = max(0, x - padding)
            y0 = max(0, y - padding)
            x1 = min(w, x + bw + padding)
            y1 = min(h, y + bh + padding)
            found.append((x0, y0, x1, y1))
        found.sort(key=lambda r: (r[1] // max(1, h // 40), r[0]))
        boxes, lines = [], []
        for i, r in enumerate(found):
            boxes.append({"rect": r, "text": "", "conf": 1.0, "index": i})
            lines.append("[%d] painted rect=%s" % (i, r))
        report = "\n".join(lines) if lines else "mask is empty - nothing painted"
        print("[TextBoxesFromMask] %d region(s)\n%s" % (len(boxes), report))
        return ({"boxes": boxes, "size": (w, h)}, report)


class ManualTextBox:
    """Define a box by hand, optionally appending to boxes from OCR."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "x": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "y": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "width": ("INT", {"default": 200, "min": 1, "max": 100000, "step": 1}),
                "height": ("INT", {"default": 60, "min": 1, "max": 100000, "step": 1}),
                "replace_existing": ("BOOLEAN", {"default": True}),
            },
            "optional": {"boxes": ("TEXT_BOXES",)},
        }

    RETURN_TYPES = ("TEXT_BOXES", "MASK", "IMAGE")
    RETURN_NAMES = ("boxes", "mask", "preview")
    FUNCTION = "make"
    CATEGORY = "text tools"

    def make(self, image, x, y, width, height, replace_existing, boxes=None):
        rgb = _to_np(image)
        h, w = rgb.shape[:2]
        x0 = max(0, min(int(x), w - 1))
        y0 = max(0, min(int(y), h - 1))
        x1 = max(x0 + 1, min(x0 + int(width), w))
        y1 = max(y0 + 1, min(y0 + int(height), h))
        existing = [] if (replace_existing or not isinstance(boxes, dict)) \
            else list(boxes.get("boxes", []))
        existing.append({"rect": (x0, y0, x1, y1), "text": "", "conf": 1.0,
                         "index": len(existing)})
        print("[ManualTextBox] box (%d,%d,%d,%d) - %d total" % (x0, y0, x1, y1, len(existing)))
        return ({"boxes": existing, "size": (w, h)},
                _boxes_to_mask(existing, w, h),
                _annotate(rgb, existing, set(range(len(existing)))))


class MoveTextBoxes:
    """Shift / resize the target regions - reposition where text gets drawn."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "boxes": ("TEXT_BOXES",),
                "dx": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1}),
                "dy": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1}),
                "scale_w": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 10.0, "step": 0.05}),
                "scale_h": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 10.0, "step": 0.05}),
                "only": ("STRING", {"default": ""}),
            },
            "optional": {"image": ("IMAGE",)},
        }

    RETURN_TYPES = ("TEXT_BOXES", "MASK", "IMAGE")
    RETURN_NAMES = ("boxes", "mask", "preview")
    FUNCTION = "move"
    CATEGORY = "text tools"

    def move(self, boxes, dx, dy, scale_w, scale_h, only, image=None):
        blist = boxes.get("boxes", []) if isinstance(boxes, dict) else []
        w, h = boxes.get("size", (0, 0)) if isinstance(boxes, dict) else (0, 0)
        if image is not None:
            rgb = _to_np(image)
            h, w = rgb.shape[:2]
        else:
            rgb = None
        idx = _parse_indices(only, len(blist))

        out = []
        for i, b in enumerate(blist):
            x0, y0, x1, y1 = b["rect"]
            if i in idx:
                cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                nw, nh = (x1 - x0) * float(scale_w), (y1 - y0) * float(scale_h)
                x0 = int(round(cx - nw / 2.0)) + int(dx)
                x1 = int(round(cx + nw / 2.0)) + int(dx)
                y0 = int(round(cy - nh / 2.0)) + int(dy)
                y1 = int(round(cy + nh / 2.0)) + int(dy)
                if w and h:
                    x0 = max(0, min(x0, w - 1)); x1 = max(x0 + 1, min(x1, w))
                    y0 = max(0, min(y0, h - 1)); y1 = max(y0 + 1, min(y1, h))
            nb = dict(b)
            nb["rect"] = (x0, y0, x1, y1)
            out.append(nb)
            print("[MoveTextBoxes] box %d -> (%d,%d,%d,%d)%s"
                  % (i, x0, y0, x1, y1, " moved" if i in idx else ""))

        prev = _annotate(rgb, out, idx) if rgb is not None else \
            torch.zeros((1, max(1, h), max(1, w), 3), dtype=torch.float32)
        return ({"boxes": out, "size": (w, h)}, _boxes_to_mask(out, w, h), prev)


NODE_CLASS_MAPPINGS = {
    "TextDetectMask": TextDetectMask,
    "DrawTextInBoxes": DrawTextInBoxes,
    "SelectTextBoxes": SelectTextBoxes,
    "TextBoxesFromMask": TextBoxesFromMask,
    "ManualTextBox": ManualTextBox,
    "MoveTextBoxes": MoveTextBoxes,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TextDetectMask": "Detect Text to Mask",
    "DrawTextInBoxes": "Draw Text In Boxes",
    "SelectTextBoxes": "Select Text Boxes",
    "TextBoxesFromMask": "Text Boxes From Painted Mask",
    "ManualTextBox": "Manual Text Box",
    "MoveTextBoxes": "Move / Resize Text Boxes",
}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
