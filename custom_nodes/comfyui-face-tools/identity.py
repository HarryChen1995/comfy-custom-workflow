"""Identity-matched face swap.

ReActor picks the target face by POSITION in a size-sorted list, so the index
means "3rd biggest face in this frame" - it does not follow a person across
cuts or crowds. These nodes pick the target by IDENTITY instead, using the
ArcFace embedding buffalo_l already computes and ReActor discards.
"""
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REACTOR_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "ComfyUI-ReActor")


def _reactor():
    if REACTOR_DIR not in sys.path:
        sys.path.insert(0, REACTOR_DIR)
    from scripts import reactor_swapper as rs
    return rs


def _to_np(image, idx=0):
    arr = image[idx].detach().cpu().numpy()
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def _to_tensor_batch(arrs):
    a = np.stack([x.astype(np.float32) / 255.0 for x in arrs], axis=0)
    return torch.from_numpy(a)


def _font(sz):
    for n in ("arialbd.ttf", "arial.ttf"):
        p = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", n)
        if os.path.isfile(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def _embed(face):
    """Unit-length ArcFace vector for a detected face."""
    e = getattr(face, "normed_embedding", None)
    if e is None:
        e = getattr(face, "embedding", None)
        if e is None:
            return None
        n = np.linalg.norm(e)
        e = e / n if n > 0 else e
    return np.asarray(e, dtype=np.float32).flatten()


def _analyze(rs, bgr, det):
    rs.ANALYSIS_MODELS.setdefault(str(int(det)), None)
    return rs.analyze_faces(bgr, det_size=(int(det), int(det)))


class FaceIdentityReference:
    """Capture WHO to replace: the embedding of one face in a reference frame."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "face_index": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "order": (["large-small", "small-large", "left-right", "right-left",
                           "top-bottom", "bottom-top"], {"default": "large-small"}),
                "det_size": ("INT", {"default": 640, "min": 160, "max": 2048, "step": 32}),
            },
            "optional": {"append_to": ("FACE_IDENTITY",)},
        }

    RETURN_TYPES = ("FACE_IDENTITY", "IMAGE", "STRING")
    RETURN_NAMES = ("identity", "preview", "report")
    FUNCTION = "run"
    CATEGORY = "face tools"

    def run(self, image, face_index, order, det_size, append_to=None):
        import cv2
        rs = _reactor()
        rgb = _to_np(image)
        faces = _analyze(rs, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), det_size)
        faces = rs.sort_by_order(faces, order) if faces else []

        prev = Image.fromarray(rgb.copy())
        d = ImageDraw.Draw(prev)
        for k, f in enumerate(faces):
            x0, y0, x1, y1 = [int(v) for v in f.bbox]
            col = (0, 230, 90) if k == face_index else (230, 70, 70)
            d.rectangle([x0, y0, x1, y1], outline=col, width=max(2, (y1 - y0) // 22))
            fo = _font(max(11, int((y1 - y0) * 0.5)))
            d.text((x0 + 3, max(0, y0 - int((y1 - y0) * 0.55))), str(k), font=fo, fill=col)

        prior = []
        if isinstance(append_to, dict):
            prior = list(append_to.get("embeddings", []))

        if not faces:
            return (append_to, _to_tensor_batch([np.array(prev)]),
                    "NO FACES in this reference frame - pick another frame")
        if face_index >= len(faces):
            return (append_to, _to_tensor_batch([np.array(prev)]),
                    "face_index %d out of range - only %d face(s) found"
                    % (face_index, len(faces)))

        emb = _embed(faces[face_index])
        if emb is None:
            return (append_to, _to_tensor_batch([np.array(prev)]), "no embedding available")

        embs = prior + [emb]
        # how different is this look from the ones already captured?
        extra = ""
        if prior:
            sims = [float(np.dot(emb, p)) for p in prior]
            extra = "  (similarity to existing refs: %s)" % ", ".join("%.2f" % s for s in sims)
        rep = ("reference %d captured: face %d of %d%s"
               % (len(embs), face_index, len(faces), extra))
        print("[FaceIdentityReference] " + rep)
        return ({"embeddings": embs, "order": order, "det_size": int(det_size)},
                _to_tensor_batch([np.array(prev)]), rep)


class FaceIdentitySwap:
    """Swap the matching person in every frame, wherever they are in the list."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "source_image": ("IMAGE",),
                "identity": ("FACE_IDENTITY",),
                "similarity_threshold": ("FLOAT", {"default": 0.35, "min": 0.0,
                                                   "max": 1.0, "step": 0.01}),
                "source_face_index": ("INT", {"default": 0, "min": 0, "max": 100}),
                "det_size": ("INT", {"default": 640, "min": 160, "max": 2048, "step": 32}),
                "swap_model": ("STRING", {"default": "inswapper_128.onnx"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE")
    RETURN_NAMES = ("images", "report", "score_plot")
    FUNCTION = "run"
    CATEGORY = "face tools"

    def run(self, images, source_image, identity, similarity_threshold,
            source_face_index, det_size, swap_model):
        import cv2
        import folder_paths
        rs = _reactor()

        if not isinstance(identity, dict) or not identity.get("embeddings"):
            raise RuntimeError("identity input is empty - the reference frame had no "
                               "usable face. Pick a different reference frame.")
        refs = [np.asarray(e, dtype=np.float32) for e in identity["embeddings"]]

        # --- source face (the one being pasted on) ---
        src_rgb = _to_np(source_image)
        src_faces = _analyze(rs, cv2.cvtColor(src_rgb, cv2.COLOR_RGB2BGR), det_size)
        src_faces = rs.sort_by_order(src_faces, "large-small") if src_faces else []
        if not src_faces:
            raise RuntimeError("no face found in source_image")
        if source_face_index >= len(src_faces):
            raise RuntimeError("source_face_index %d but source has %d face(s)"
                               % (source_face_index, len(src_faces)))
        source_face = src_faces[source_face_index]

        model_path = folder_paths.get_full_path("insightface", swap_model)
        if model_path is None:
            model_path = os.path.join(folder_paths.models_dir, "insightface", swap_model)
        swapper = rs.getFaceSwapModel(model_path)

        n = images.shape[0]
        out, scores, swapped = [], [], 0
        for i in range(n):
            rgb = _to_np(images, i)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            faces = _analyze(rs, bgr, det_size)
            best, best_s = None, -1.0
            for f in (faces or []):
                e = _embed(f)
                if e is None:
                    continue
                # best match across every captured look of this person
                s = max(float(np.dot(e, r)) for r in refs if r.shape == e.shape) \
                    if any(r.shape == e.shape for r in refs) else -1.0
                if s > best_s:
                    best_s, best = s, f
            scores.append(best_s if best is not None else -1.0)
            if best is not None and best_s >= float(similarity_threshold):
                res = swapper.get(bgr, best, source_face, paste_back=True)
                out.append(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
                swapped += 1
            else:
                out.append(rgb)                     # person not here - leave frame alone
            if (i + 1) % 25 == 0 or i == n - 1:
                print("[FaceIdentitySwap] %d/%d frames, %d swapped"
                      % (i + 1, n, swapped), flush=True)

        arr = np.array([s for s in scores if s >= 0], dtype=np.float32)
        rep = ("%d frames | %d swapped (%.0f%%) | threshold %.2f\n"
               % (n, swapped, 100.0 * swapped / max(1, n), similarity_threshold))
        if arr.size:
            rep += ("match score: min %.2f  median %.2f  max %.2f  "
                    "(frames with a face: %d)"
                    % (arr.min(), float(np.median(arr)), arr.max(), arr.size))
        else:
            rep += "no faces detected in any frame"
        print("[FaceIdentitySwap] " + rep)
        return (_to_tensor_batch(out), rep, self._plot(scores, similarity_threshold))

    def _plot(self, scores, thr, W=900, H=260):
        """Per-frame similarity so the threshold can be set by eye, not guessed."""
        img = Image.new("RGB", (W, H), (24, 24, 28))
        d = ImageDraw.Draw(img)
        l, r, t, b = 50, W - 12, 18, H - 30
        f = _font(13)
        for v in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = b - (b - t) * v
            d.line([(l, y), (r, y)], fill=(58, 58, 66))
            d.text((6, y - 7), "%.2f" % v, font=f, fill=(150, 150, 160))
        n = max(1, len(scores))
        pts = []
        for i, s in enumerate(scores):
            x = l + (r - l) * (i / max(1, n - 1))
            pts.append((x, b - (b - t) * max(0.0, min(1.0, s))))
        ty = b - (b - t) * max(0.0, min(1.0, thr))
        d.line([(l, ty), (r, ty)], fill=(255, 90, 90), width=2)
        d.text((l + 4, ty - 16), "threshold %.2f" % thr, font=f, fill=(255, 120, 120))
        for i in range(1, len(pts)):
            if scores[i] < 0 or scores[i - 1] < 0:
                continue
            col = (90, 230, 130) if scores[i] >= thr else (235, 190, 80)
            d.line([pts[i - 1], pts[i]], fill=col, width=2)
        for i, s in enumerate(scores):
            if s < 0:
                x = l + (r - l) * (i / max(1, n - 1))
                d.line([(x, b), (x, b - 6)], fill=(120, 120, 130))
        d.text((l, H - 18), "frame 0", font=f, fill=(150, 150, 160))
        d.text((r - 70, H - 18), "frame %d" % (n - 1), font=f, fill=(150, 150, 160))
        d.text((l + 110, H - 18), "green = swapped   amber = below threshold   "
                                  "grey ticks = no face", font=f, fill=(150, 150, 160))
        return _to_tensor_batch([np.array(img)])


NODE_CLASS_MAPPINGS = {
    "FaceIdentityReference": FaceIdentityReference,
    "FaceIdentitySwap": FaceIdentitySwap,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceIdentityReference": "Face Identity Reference (lock onto a person)",
    "FaceIdentitySwap": "Face Identity Swap (follows the person)",
}
