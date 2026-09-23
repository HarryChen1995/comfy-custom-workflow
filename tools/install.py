#!/usr/bin/env python3
"""Install this repo's custom nodes and workflows into ComfyUI.

Two modes:

    python tools/install.py                # LINK  (default) - edits flow both ways
    python tools/install.py --copy         # COPY  - snapshot, repo and ComfyUI independent

Link mode uses directory junctions on Windows (no admin needed) and symlinks
elsewhere, so a workflow you save in ComfyUI lands straight in your git working
tree. Copy mode is the safe choice if you would rather keep them separate.

    python tools/install.py --comfy D:\\AI\\ComfyUI   # if auto-detect fails
    python tools/install.py --uninstall              # remove links, restore copies
    python tools/install.py --check                  # report current state only
"""
import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODE_DIRS = ["comfyui-face-tools", "comfyui-text-tools"]
WIN = os.name == "nt"


def find_comfy(explicit=None):
    cands = []
    if explicit:
        cands.append(explicit)
    if os.environ.get("COMFYUI_DIR"):
        cands.append(os.environ["COMFYUI_DIR"])
    cands += [
        os.path.dirname(HERE),
        os.path.join(os.path.expanduser("~"), "ComfyUI"),
        r"C:\ComfyUI",
        os.path.join(os.getcwd(), "ComfyUI"),
        os.getcwd(),
    ]
    for c in cands:
        if c and os.path.isdir(os.path.join(c, "models")) and \
                os.path.isfile(os.path.join(c, "main.py")):
            return os.path.abspath(c)
    return None


def is_link(p):
    if os.path.islink(p):
        return True
    if WIN and os.path.isdir(p):
        try:                                  # junctions are reparse points
            return bool(os.readlink(p))
        except OSError:
            return False
    return False


def link_dir(target, link):
    """Junction on Windows (no admin), symlink elsewhere."""
    if WIN:
        r = subprocess.run(["cmd", "/c", "mklink", "/J", link, target],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip() or r.stdout.strip())
    else:
        os.symlink(target, link)


def backup(path, bdir):
    if not os.path.exists(path) or is_link(path):
        return None
    os.makedirs(bdir, exist_ok=True)
    dest = os.path.join(bdir, os.path.basename(path.rstrip("/\\")))
    if os.path.exists(dest):
        shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(path, dest)
    return dest


def plan(root):
    wf = os.path.join(root, "user", "default", "workflows")
    items = [(os.path.join(HERE, "workflows"), wf)]
    for n in NODE_DIRS:
        items.append((os.path.join(HERE, "custom_nodes", n),
                      os.path.join(root, "custom_nodes", n)))
    return items


def do_check(root):
    print("ComfyUI: %s\n" % root)
    for src, dst in plan(root):
        if not os.path.exists(dst):
            state = "MISSING"
        elif is_link(dst):
            try:
                tgt = os.readlink(dst)
            except OSError:
                tgt = "?"
            state = "LINK -> %s" % tgt
        else:
            state = "copy (%d files)" % sum(len(f) for _, _, f in os.walk(dst))
        print("  %-52s %s" % (os.path.relpath(dst, root), state))
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comfy", help="path to ComfyUI (only if auto-detect fails)")
    ap.add_argument("--copy", action="store_true", help="copy instead of link")
    ap.add_argument("--uninstall", action="store_true", help="remove links")
    ap.add_argument("--check", action="store_true", help="report state and exit")
    a = ap.parse_args()

    root = find_comfy(a.comfy)
    if not root:
        print("Could not find ComfyUI. Pass it:  --comfy C:\\ComfyUI")
        return 2

    if a.check:
        return do_check(root)

    bdir = os.path.join(root, "_backup_before_install")

    if a.uninstall:
        for _src, dst in plan(root):
            if is_link(dst):
                if WIN and os.path.isdir(dst):
                    subprocess.run(["cmd", "/c", "rmdir", dst], capture_output=True)
                else:
                    os.unlink(dst)
                print("[unlink] %s" % os.path.relpath(dst, root))
            else:
                print("[keep  ] %s (not a link)" % os.path.relpath(dst, root))
        print("\nBackups, if any, are in %s" % bdir)
        return 0

    print("ComfyUI: %s" % root)
    print("mode   : %s\n" % ("COPY" if a.copy else "LINK"))

    for src, dst in plan(root):
        rel = os.path.relpath(dst, root)
        os.makedirs(os.path.dirname(dst), exist_ok=True)

        if a.copy:
            os.makedirs(dst, exist_ok=True)
            n = 0
            for name in os.listdir(src):
                if name == "__pycache__":
                    continue
                s = os.path.join(src, name)
                d = os.path.join(dst, name)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)
                n += 1
            print("[copy  ] %-50s %d item(s)" % (rel, n))
            continue

        if is_link(dst):
            print("[ok    ] %-50s already linked" % rel)
            continue
        b = backup(dst, bdir)
        if b:
            print("[backup] %-50s -> %s" % (rel, os.path.relpath(b, root)))
            shutil.rmtree(dst, ignore_errors=True)
        elif os.path.exists(dst):
            shutil.rmtree(dst, ignore_errors=True)
        try:
            link_dir(src, dst)
            print("[link  ] %-50s -> %s" % (rel, src))
        except Exception as e:
            print("[FAIL  ] %s: %s" % (rel, e))
            if b:
                shutil.copytree(b, dst)
                print("         restored from backup")
            return 1

    print("""
Next:
  1. pip install -r requirements.txt   (into ComfyUI's Python)
  2. python tools/download_models.py
  3. RESTART ComfyUI - custom node code only loads at startup
  4. The workflows appear in the sidebar Workflows tab""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
