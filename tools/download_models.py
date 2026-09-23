#!/usr/bin/env python3
"""Download every model these workflows need.

Just run it - no arguments:

    python tools/download_models.py

It looks for your ComfyUI folder automatically (repo parent, C:\\ComfyUI,
~/ComfyUI, $COMFYUI_DIR, ...). Override only if it guesses wrong:

    python tools/download_models.py --comfy D:\\AI\\ComfyUI
    python tools/download_models.py --list        # show the list, download nothing
    python tools/download_models.py --skip sd     # everything except SDXL Turbo

Files already present with the right byte count are skipped. Interrupted
downloads resume from where they stopped, so just run it again.
"""
import argparse
import os
import sys
import urllib.request

MODELS = [
    dict(group="face", name="inswapper_128.onnx", sub="insightface",
         url="https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/inswapper_128.onnx",
         size=554253681,
         note="ReActor face swap. InsightFace models: NON-COMMERCIAL / research use only."),
    dict(group="face", name="GFPGANv1.4.pth", sub="facerestore_models",
         url="https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GFPGANv1.4.pth",
         size=348632874,
         note="Face restoration after swapping. Apache-2.0."),
    dict(group="text", name="big-lama.pt", sub="inpaint",
         url="https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
         size=205669692,
         note="LaMa inpainting - erases text and objects."),
    dict(group="sd", name="sd_xl_turbo_1.0_fp16.safetensors", sub="checkpoints",
         url="https://huggingface.co/stabilityai/sdxl-turbo/resolve/main/sd_xl_turbo_1.0_fp16.safetensors",
         size=6938081905,
         note="SDXL Turbo - only for plain text-to-image, not needed by the "
              "face or text workflows. Skip with --skip sd."),
]

AUTO = """
Fetched automatically on first use, nothing to do:
  buffalo_l/*        face detection + ArcFace embeddings -> models/insightface/models/
  EasyOCR models     text detection -> ~/.EasyOCR/
"""


def human(n):
    if n is None:
        return "?"
    v = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if v < 1024 or unit == "GB":
            return "%.1f %s" % (v, unit)
        v /= 1024.0


def find_comfy(explicit=None):
    """Locate a ComfyUI install: a folder containing models/ and main.py."""
    cands = []
    if explicit:
        cands.append(explicit)
    if os.environ.get("COMFYUI_DIR"):
        cands.append(os.environ["COMFYUI_DIR"])
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cands += [
        os.path.dirname(here),                  # repo sitting inside ComfyUI
        here,
        os.path.join(os.path.expanduser("~"), "ComfyUI"),
        r"C:\ComfyUI",
        os.path.join(os.getcwd(), "ComfyUI"),
        os.getcwd(),
    ]
    for c in cands:
        if c and os.path.isdir(os.path.join(c, "models")) and \
                os.path.isfile(os.path.join(c, "main.py")):
            return os.path.abspath(c)
    for c in cands:                             # models/ alone is good enough
        if c and os.path.isdir(os.path.join(c, "models")):
            return os.path.abspath(c)
    return None


def fetch(url, dest, expect):
    tmp = dest + ".part"
    have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    if expect and have >= expect:
        have = 0
        os.remove(tmp)
    req = urllib.request.Request(url, headers={"User-Agent": "comfy-custom-workflow"})
    if have:
        req.add_header("Range", "bytes=%d-" % have)
        print("    resuming at %s" % human(have))
    with urllib.request.urlopen(req, timeout=60) as r:
        # if the server ignored our Range, start over
        if have and r.status != 206:
            have = 0
        mode = "ab" if have else "wb"
        total = r.headers.get("Content-Length")
        total = (int(total) + have) if total else expect
        done = have
        with open(tmp, mode) as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    sys.stdout.write("\r    %5.1f%%  %s / %s"
                                     % (100.0 * done / total, human(done), human(total)))
                else:
                    sys.stdout.write("\r    %s" % human(done))
                sys.stdout.flush()
    sys.stdout.write("\n")
    os.replace(tmp, dest)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comfy", help="path to ComfyUI (only if auto-detect fails)")
    ap.add_argument("--skip", action="append", default=[],
                    choices=sorted({m["group"] for m in MODELS}),
                    help="skip a group: face, text, sd  (repeatable)")
    ap.add_argument("--list", action="store_true", help="show the list and exit")
    a = ap.parse_args()

    want = [m for m in MODELS if m["group"] not in a.skip]

    if a.list:
        total = sum(m["size"] for m in want if m["size"])
        for m in want:
            print("[%-4s] %-36s %10s  -> models/%s" %
                  (m["group"], m["name"], human(m["size"]), m["sub"]))
            print("        %s" % m["note"])
        print("\ntotal: %s" % human(total))
        print(AUTO)
        return 0

    root = find_comfy(a.comfy)
    if not root:
        print("Could not find your ComfyUI folder.")
        print("Pass it explicitly:  python tools/download_models.py --comfy C:\\ComfyUI")
        return 2
    print("ComfyUI: %s\n" % root)

    failed = []
    for m in want:
        d = os.path.join(root, "models", m["sub"])
        os.makedirs(d, exist_ok=True)
        dest = os.path.join(d, m["name"])
        if os.path.exists(dest):
            have = os.path.getsize(dest)
            if m["size"] is None or have == m["size"]:
                print("[skip] %-36s present (%s)" % (m["name"], human(have)))
                continue
            print("[warn] %s is %s, expected %s - re-downloading"
                  % (m["name"], human(have), human(m["size"])))
            os.remove(dest)
        print("[get ] %-36s -> models/%s" % (m["name"], m["sub"]))
        try:
            fetch(m["url"], dest, m["size"])
        except Exception as e:
            print("    FAILED: %s" % e)
            failed.append(m["name"])
            continue
        got = os.path.getsize(dest)
        if m["size"] and got != m["size"]:
            print("    SIZE MISMATCH: got %s, expected %s" % (human(got), human(m["size"])))
            failed.append(m["name"])
        else:
            print("    ok (%s)" % human(got))

    print(AUTO)
    if failed:
        print("FAILED: %s\nRe-run to resume." % ", ".join(failed))
        return 1
    print("All models present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
