"""Multi-person identity-matched swap.

FaceIdentityPair      - bind "this person in the video" to "that face in my photo"
FaceIdentityMultiSwap - per frame, assign each detected face to the best pair

Assignment is greedy on the score matrix: the highest (face, person) score is
taken first, then that face AND that person are removed from consideration, so
one frame can never give two people the same face or one person two faces.
"""
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .identity import (_reactor, _to_np, _to_tensor_batch, _embed, _analyze, _font)

LINE_COLORS = [(90, 230, 130), (255, 190, 70), (110, 190, 255),
               (255, 120, 200), (180, 140, 255), (255, 140, 110)]


class FaceIdentityPair:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "identity": ("FACE_IDENTITY",),
                "source_photo_index": ("INT", {"default": 0, "min": 0, "max": 100}),
                "label": ("STRING", {"default": "person A"}),
            },
            "optional": {"append_to": ("FACE_PAIRS",)},
        }

    RETURN_TYPES = ("FACE_PAIRS", "STRING")
    RETURN_NAMES = ("pairs", "report")
    FUNCTION = "run"
    CATEGORY = "face tools"

    def run(self, identity, source_photo_index, label, append_to=None):
        prior = list(append_to.get("pairs", [])) if isinstance(append_to, dict) else []
        if not isinstance(identity, dict) or not identity.get("embeddings"):
            rep = "identity is empty - that reference frame had no usable face"
            print("[FaceIdentityPair] " + rep)
            return ({"pairs": prior}, rep)
        prior.append({"embeddings": [np.asarray(e, dtype=np.float32)
                                     for e in identity["embeddings"]],
                      "source_photo_index": int(source_photo_index),
                      "label": label.strip() or ("person %d" % (len(prior) + 1))})
        lines = ["[%d] %-14s <- source photo %d   (%d reference look(s))"
                 % (k, p["label"], p["source_photo_index"], len(p["embeddings"]))
                 for k, p in enumerate(prior)]
        rep = "%d pair(s)\n%s" % (len(prior), "\n".join(lines))
        print("[FaceIdentityPair] " + rep)
        return ({"pairs": prior}, rep)


class FaceIdentityMultiSwap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "source_set": ("SOURCE_SET",),
                "pairs": ("FACE_PAIRS",),
                "similarity_threshold": ("FLOAT", {"default": 0.30, "min": 0.0,
                                                   "max": 1.0, "step": 0.01}),
                "det_size": ("INT", {"default": 640, "min": 160, "max": 2048, "step": 32}),
                "swap_model": ("STRING", {"default": "inswapper_128.onnx"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE")
    RETURN_NAMES = ("images", "report", "score_plot")
    FUNCTION = "run"
    CATEGORY = "face tools"

    def run(self, images, source_set, pairs, similarity_threshold, det_size, swap_model):
        import cv2
        import folder_paths
        from .sources import _pick_face
        rs = _reactor()

        plist = pairs.get("pairs", []) if isinstance(pairs, dict) else []
        if not plist:
            raise RuntimeError("no pairs supplied - chain at least one Face Identity Pair")

        photos = source_set.get("photos", []) if isinstance(source_set, dict) else []
        if not photos:
            raise RuntimeError("source_set is empty - chain at least one Source Photo Set")

        # resolve each pair's source photo to an actual face, once
        src_faces = {}
        for p in plist:
            pi = p["source_photo_index"]
            if not (0 <= pi < len(photos)):
                raise RuntimeError("pair %r wants source photo %d but only %d loaded"
                                   % (p["label"], pi, len(photos)))
            if pi not in src_faces:
                ph = photos[pi]
                f, _n = _pick_face(rs, ph["rgb"], ph["order"], ph["det_size"],
                                   ph["face_index"])
                if f is None:
                    raise RuntimeError("source photo %d (%s) has no usable face"
                                       % (pi, ph["label"]))
                src_faces[pi] = f

        model_path = folder_paths.get_full_path("insightface", swap_model)
        if model_path is None:
            model_path = os.path.join(folder_paths.models_dir, "insightface", swap_model)
        swapper = rs.getFaceSwapModel(model_path)

        thr = float(similarity_threshold)
        n = images.shape[0]
        npair = len(plist)
        out = []
        per_pair = [[] for _ in range(npair)]      # best score per frame, per person
        counts = [0] * npair
        frames_swapped = 0

        for fi in range(n):
            rgb = _to_np(images, fi)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            faces = _analyze(rs, bgr, det_size) or []
            embs = [(f, _embed(f)) for f in faces]
            embs = [(f, e) for f, e in embs if e is not None]

            # score matrix: detected face x person
            grid = np.full((len(embs), npair), -1.0, dtype=np.float32)
            for a, (_, e) in enumerate(embs):
                for b, p in enumerate(plist):
                    ok = [r for r in p["embeddings"] if r.shape == e.shape]
                    if ok:
                        grid[a, b] = max(float(np.dot(e, r)) for r in ok)

            for b in range(npair):
                col = grid[:, b]
                per_pair[b].append(float(col.max()) if col.size else -1.0)

            # greedy one-to-one assignment
            did = False
            used_f, used_p = set(), set()
            work = grid.copy()
            while True:
                if work.size == 0:
                    break
                a, b = np.unravel_index(int(np.argmax(work)), work.shape)
                if work[a, b] < thr:
                    break
                sf = src_faces[plist[b]["source_photo_index"]]
                bgr = swapper.get(bgr, embs[a][0], sf, paste_back=True)
                counts[b] += 1
                did = True
                used_f.add(a)
                used_p.add(b)
                work[a, :] = -1.0
                work[:, b] = -1.0
            out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if did else rgb)
            if did:
                frames_swapped += 1
            if (fi + 1) % 25 == 0 or fi == n - 1:
                print("[FaceIdentityMultiSwap] %d/%d frames, %d touched"
                      % (fi + 1, n, frames_swapped), flush=True)

        lines = ["%d frames | %d had at least one swap | threshold %.2f"
                 % (n, frames_swapped, thr)]
        for b, p in enumerate(plist):
            arr = np.array([s for s in per_pair[b] if s >= 0], dtype=np.float32)
            if arr.size:
                lines.append("  [%d] %-14s swapped in %4d frames | score med %.2f max %.2f"
                             % (b, p["label"], counts[b], float(np.median(arr)), arr.max()))
            else:
                lines.append("  [%d] %-14s never matched" % (b, p["label"]))
        rep = "\n".join(lines)
        print("[FaceIdentityMultiSwap] " + rep)
        return (_to_tensor_batch(out), rep, self._plot(per_pair, plist, thr))

    def _plot(self, per_pair, plist, thr, W=980, H=300):
        img = Image.new("RGB", (W, H), (24, 24, 28))
        d = ImageDraw.Draw(img)
        l, r, t, b = 52, W - 12, 18, H - 46
        f = _font(13)
        for v in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = b - (b - t) * v
            d.line([(l, y), (r, y)], fill=(58, 58, 66))
            d.text((6, y - 7), "%.2f" % v, font=f, fill=(150, 150, 160))
        ty = b - (b - t) * max(0.0, min(1.0, thr))
        d.line([(l, ty), (r, ty)], fill=(255, 90, 90), width=2)
        d.text((l + 4, ty - 16), "threshold %.2f" % thr, font=f, fill=(255, 120, 120))
        n = max(1, len(per_pair[0]) if per_pair else 1)
        for bi, series in enumerate(per_pair):
            col = LINE_COLORS[bi % len(LINE_COLORS)]
            prev = None
            for i, s in enumerate(series):
                if s < 0:
                    prev = None
                    continue
                x = l + (r - l) * (i / max(1, n - 1))
                y = b - (b - t) * max(0.0, min(1.0, s))
                if prev is not None:
                    d.line([prev, (x, y)], fill=col, width=2)
                prev = (x, y)
            d.rectangle([l + bi * 150, H - 34, l + bi * 150 + 14, H - 22], fill=col)
            d.text((l + bi * 150 + 20, H - 36), plist[bi]["label"][:16], font=f,
                   fill=(210, 210, 215))
        d.text((l, H - 16), "one line per person - gaps = that person not detected",
               font=f, fill=(150, 150, 160))
        return _to_tensor_batch([np.array(img)])


NODE_CLASS_MAPPINGS = {
    "FaceIdentityPair": FaceIdentityPair,
    "FaceIdentityMultiSwap": FaceIdentityMultiSwap,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceIdentityPair": "Face Identity Pair (person <- source face)",
    "FaceIdentityMultiSwap": "Face Identity Multi Swap",
}
