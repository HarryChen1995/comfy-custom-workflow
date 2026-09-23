"""Validate every saved workflow against the live server schema.

Handles BOTH serialisation formats:
  - old: widgets only in widgets_values, inputs = real links
  - new: widgets ALSO mirrored into inputs[] with a 'widget' key
and checks each widget value is type-compatible with its declared slot.
"""
import glob, json, os, urllib.request

oi = json.load(urllib.request.urlopen("http://127.0.0.1:8188/object_info", timeout=60))

SKIP = {"LoadImage", "SaveImage", "PreviewImage", "MarkdownNote", "Note", "Reroute"}


def typename(spec):
    t = spec[0]
    if isinstance(t, list):
        return "COMBO", t
    if isinstance(t, str) and t == "COMBO":
        opts = spec[1].get("options", []) if len(spec) > 1 and isinstance(spec[1], dict) else []
        return "COMBO", opts
    return t, None


def compatible(val, tname, opts):
    if tname == "COMBO":
        return (not opts) or (val in opts)
    if tname == "INT":
        return isinstance(val, int) and not isinstance(val, bool)
    if tname == "FLOAT":
        return isinstance(val, (int, float)) and not isinstance(val, bool)
    if tname == "BOOLEAN":
        return isinstance(val, bool)
    if tname == "STRING":
        return isinstance(val, str)
    return True


problems = 0
for path in sorted(glob.glob(r"C:\ComfyUI\user\default\workflows\*.json")):
    name = os.path.basename(path)
    wf = json.load(open(path, encoding="utf-8"))
    bad = []
    for nd in wf.get("nodes", []):
        t = nd.get("type")
        if t in SKIP:
            continue
        if t not in oi:
            bad.append("unknown node type: %s" % t)
            continue
        req = oi[t]["input"].get("required", {})
        opt = oi[t]["input"].get("optional", {})
        order = oi[t].get("input_order", {}).get("required", list(req.keys()))
        opt_order = oi[t].get("input_order", {}).get("optional", list(opt.keys()))
        # a name is a real LINK only if it has no 'widget' marker
        linked = {x["name"] for x in nd.get("inputs", []) if not x.get("widget")}
        wnames = [k for k in order if k not in linked]
        # optional PRIMITIVE inputs are widgets too and are serialised as well
        WIDGETY = ("INT", "FLOAT", "STRING", "BOOLEAN", "COMBO")
        for k in opt_order:
            if k in linked:
                continue
            tn = typename(opt[k])[0]
            if tn in WIDGETY or (isinstance(tn, str) and "COMBO" in tn):
                wnames.append(k)
        req = dict(req)
        req.update(opt)
        wv = nd.get("widgets_values", [])

        # frontend appends control_after_generate right after a seed-ish INT
        expanded = []
        for k in wnames:
            expanded.append(k)
            spec = req.get(k)
            if spec and typename(spec)[0] == "INT" and k in ("seed", "noise_seed"):
                expanded.append("__control__")

        # The frontend appends its own trailing values (upload-widget mode on
        # LoadImage/LoadVideo, dynamic-combo expansion on SaveVideo). Extra
        # trailing entries are fine; too FEW means the node is genuinely stale.
        if len(wv) < len(wnames):
            bad.append("%s: %d widget values, expected at least %d (%s)"
                       % (t, len(wv), len(wnames), wnames))
            continue

        names = expanded if len(wv) >= len(expanded) else wnames
        for val, k in zip(wv, names):
            if k == "__control__":
                continue
            spec = req.get(k)
            if not spec:
                continue
            tn, opts = typename(spec)
            if not compatible(val, tn, opts):
                bad.append("%s.%s = %r is not a valid %s%s"
                           % (t, k, val, tn,
                              (" (options: %s)" % opts[:6]) if tn == "COMBO" and opts else ""))
    if bad:
        problems += 1
        print("PROBLEM  %s" % name)
        for b in bad:
            print("     ", b)
    else:
        print("OK       %s" % name)

print()
print("ALL CLEAN" if problems == 0 else "%d workflow(s) with problems" % problems)
