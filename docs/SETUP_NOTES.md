# ComfyUI setup notes

Installed 2026-09-16. Location: `C:\ComfyUI`

## Environment
- venv: `C:\ComfyUI\venv` (Python 3.11.9) — isolated; the global Python's torch 2.6.0+cu124 is untouched
- torch 2.14.0+cu126 / torchvision 0.29 / torchaudio 2.11
- CUDA toolkit 12.6, driver 561.17

## GPUs — important
`nvidia-smi` orders by PCI slot; PyTorch orders fastest-first. They disagree:

| Card        | nvidia-smi | PyTorch / ComfyUI |
|-------------|-----------|-------------------|
| RTX 2060 6GB| 1         | **0**             |
| GTX 1060 3GB| 0         | 1                 |

`run_gpu.bat` pins `--cuda-device 0` = the RTX 2060. Correct.

## Launch
    C:\ComfyUI\run_gpu.bat
Then http://127.0.0.1:8188

Only ONE instance at a time — a second one fails with
"Database is locked. Another ComfyUI process is already using this database."

## Models
    models\checkpoints\sd_xl_turbo_1.0_fp16.safetensors   SDXL Turbo
    models\insightface\inswapper_128.onnx                 ReActor face swap
    models\facerestore_models\GFPGANv1.4.pth              face restoration
    models\inpaint\big-lama.pt                            LaMa erase/inpaint

## Custom nodes
- `ComfyUI-ReActor` v0.7.0-b1 — face swap (15 nodes). Current version needs
  NO insightface and NO C++ build tools, and is numpy 2.x compatible.
- `comfyui-inpaint-nodes` — LaMa/MAT/Fooocus inpainting (13 nodes)

## Workflows  (user\default\workflows\)
- `ReActor_MultiFace_Chained.json`    — 2 different people -> 2 faces in a group photo
- `ReActor_MultiFace_SingleNode.json` — one source photo containing several faces
- `RemoveText_LaMa.json`              — erase text/watermark/objects, no prompt needed

## Multi-face swap: how the indices work
Faces are numbered left-to-right (`large-small` ordering by default).
- `input_faces_index`  = which face(s) in the TARGET get replaced
- `source_faces_index` = which face(s) in the SOURCE are used

Both accept comma-separated lists, matched pairwise:
    input_faces_index = 0,1,2
    source_faces_index = 0,1,2
For faces coming from DIFFERENT source images, chain one ReActor node per
person instead (see the Chained workflow) — feed each node's SWAPPED_IMAGE
into the next node's input_image.

## Gotchas hit during install
- setuptools 84 removed `pkg_resources`; ReActor's install.py needs
  `importlib_metadata` installed first.
- ReActor's install.py exits before installing `ultralytics` — install manually.
- Don't run two downloads to the same file; it silently corrupts it.

## Licence note
inswapper_128.onnx and buffalo_l are InsightFace models, licensed
NON-COMMERCIAL / research use only. GFPGAN is Apache-2.0.

---

## Automatic text detect / erase / retype  (added later)

Custom node pack: `custom_nodes\comfyui-text-tools` (uses EasyOCR)

### Detect Text to Mask
| field | meaning |
|-------|---------|
| languages | e.g. `en`, or `en,fr` |
| confidence | drop detections below this (0.30 default) |
| padding | grow each box outward, px |
| **select** | which boxes: blank = ALL, `0`, `0,2`, `1-3` |
| **contains** | only boxes whose text contains this substring |
| invert_selection | act on everything EXCEPT the selection |

Outputs: `mask`, `boxes`, `detected_text`, `preview`
The **preview** output draws numbered boxes - green = selected, red = ignored.
Run once, look at the preview, then type the numbers you want into `select`.

### Draw Text In Boxes
| field | meaning |
|-------|---------|
| replacement_text | one line per selected box |
| font | any font in C:\Windows\Fonts - pick `*bd.ttf` for bold weight |
| **color** | `auto` (samples original glyph colour), `#ff3030`, `red`, or per-box `red,blue` |
| **font_size** | 0 = auto-fit to the box, else exact px |
| **bold** | faux-bold thickness, 0-30 |
| fill_ratio | how much of the box the text fills (0.92 default) |
| align | center / left / right |
| outline_width + outline_color | outline around glyphs |

Weight comes from three places: the font file (`arialbd.ttf` vs `arial.ttf`),
`bold` for faux thickening, and `outline_width` for an outline.

## Workflows
    RemoveText_Auto.json  - OCR -> mask -> LaMa erase          (no painting)
    EditText_Auto.json    - OCR -> erase -> retype             (no painting)
    RemoveText_LaMa.json  - manual MaskEditor painting version
    ReActor_MultiFace_Chained.json / _SingleNode.json

## Why "Image and mask must be the same size ... != torch.Size([64, 64])"
`LoadImage` returns a 64x64 zero placeholder mask when the image has no alpha
channel (nodes.py:1790) - i.e. nothing was painted. Either paint in MaskEditor
AND save to the node, or use the Auto workflows which build the mask by OCR.

## IMPORTANT: when changes don't show up
- New/edited **custom node python** -> RESTART the server. F5 does NOT reload it.
- New **workflow .json** -> browser refresh (F5) is enough.
- New **model file** -> refresh node definitions (R) or restart.

---

## Selecting WHICH box to edit  (not all of them)

Workflow: `EditText_SelectBox.json`

    LoadImage
      -> Detect Text to Mask      select = BLANK  (find everything)
      -> Select Text Boxes        select = 0      (<- pick what to edit)
           |-> preview  : green = will be edited, red = left alone
           |-> mask     : erase mask for ONLY the picked boxes
      -> Move / Resize Text Boxes dx / dy / scale (optional reposition)
      -> Expand Mask -> LaMa      erases only the picked boxes
      -> Draw Text In Boxes       retypes into the (moved) boxes
      -> SaveImage

Run it once, look at the preview to read the box numbers, then type them
into `Select Text Boxes -> select`. Everything not picked is untouched.

### Select Text Boxes
| field | meaning |
|-------|---------|
| select | `0`, `0,2`, `1-3`; blank = all |
| contains | only boxes whose text contains this |
| invert_selection | everything EXCEPT the selection |

Two separate selections are possible: `Detect Text to Mask -> select`
controls what is *detected*, `Select Text Boxes -> select` controls what is
*edited*. Erase the original position, draw at the moved position.

### Move / Resize Text Boxes  (reposition)
| field | meaning |
|-------|---------|
| dx / dy | shift the region, px (negative = left/up) |
| scale_w / scale_h | grow or shrink the region about its centre |
| only | which boxes to move; blank = all |

### Draw Text In Boxes - positioning
| field | meaning |
|-------|---------|
| align | center / left / right within the box |
| valign | middle / top / bottom |
| offset_x / offset_y | nudge the text, px |

Erase position comes from the mask; draw position comes from the boxes.
Feed the mask from BEFORE `Move` and the boxes from AFTER it to erase in the
old spot and retype in the new one.

### When OCR misses text
- `Text Boxes From Painted Mask` - paint in MaskEditor, get a box from it
- `Manual Text Box` - type x / y / width / height directly
Both output TEXT_BOXES, so they drop straight into the same chain.

---

## Multi-face swap with MANUAL mapping

Node pack: `custom_nodes\comfyui-face-tools`
Workflow:  `ReActor_FaceMapping.json`

    SOURCE image -> Face Detect Preview -> boxes preview + numbered face grid
    TARGET image -> Face Detect Preview -> boxes preview + numbered face grid
                         |                        |
                         +--> Face Mapping <------+
                                   |
                         source_faces_index / input_faces_index
                                   |
                              ReActorFaceSwap -> SaveImage

### Face Detect Preview  (numbered)
Uses ReActor's OWN `analyze_faces` + `sort_by_order`, so the numbers drawn are
exactly the indices ReActor uses when swapping. Do not substitute another
detector or the numbering will not line up.

| output | what it is |
|--------|------------|
| preview | original image with a numbered, colour-coded box per face |
| face_grid | contact sheet of cropped faces, numbered - use this when faces are small |
| faces | FACE_BOXES for the mapping node |
| report | text list of every face: rect, size, gender, age |
| count | number of faces found |

| field | meaning |
|-------|---------|
| order | face numbering order. DEFAULT `large-small` = biggest face is 0. This matches ReActor's default - changing it here without changing it in ReActor will mis-map. |
| det_size | detector resolution; raise to 1024 if small faces are missed |
| label_scale | label size multiplier (labels auto-scale per face) |
| show_details | show gender / estimated age |

### Face Mapping (source -> target)
One pair per line, `sourceFace > targetFace`:

    0>2
    5>8

Also accepts `0:2` and comma separated on one line.
Outputs `source_faces_index` = "0,5" and `input_faces_index` = "2,8",
wired straight into ReActorFaceSwap.

**Source and target face counts do NOT need to match.** Map only the pairs you
want; every unmapped target face is left alone. One source face may be mapped
onto several target faces (`0>1`, `0>4`). Mapping the same TARGET twice is
rejected - the first wins - and out-of-range indices are reported as warnings
in the node's `report` output (wired to a PreviewAny node).

### Tested
avengers.jpg (375x533) -> 19 faces detected. Mapping `0>2, 5>8` put Thanos's
face on target 2 and Black Widow's on target 8; all other faces untouched.
Detection ~12 s, full swap ~33 s on the RTX 2060.

---

## Image size: what actually works  (tested on the RTX 2060, 6 GB)

Every workflow ran successfully at every size tried. Nothing OOM'd.

### LaMa text erase
| image | megapixels | time |
|-------|-----------|------|
| 375x533    | 0.2  | 9 s  |
| 2250x3198  | 7.2  | 6 s  |
| 3750x5330  | 20.0 | 8 s  |
| 5250x7462  | 39.2 | 12 s |

No practical ceiling - 39 MP is far beyond any normal photo.

### Face swap (19 faces, 2 swapped, GFPGAN restore on)
| image | time |
|-------|------|
| 375x533   | 33 s |
| 1125x1599 | 44 s |
| 2250x3198 | 60 s |

### THE CATCH: face numbering shifts with resolution
The detector finds a DIFFERENT NUMBER of faces at different sizes, so the
indices are not portable between resolutions:

| image size | det_size | faces found |
|-----------|----------|-------------|
| 187x266   | 640 auto | **15** |
| 375x533   | 640 auto | **19** |
| 750x1066  | 640 auto | **19** |
| 1125x1599 | 800 auto | **20** |
| 2250x3198 | 1280 auto| **19** |
| 375x533   | 320      | **10** |
| 1125x1599 | 1024     | **18** |

Consequences:
- Read the numbers and do the swap in the SAME run at the SAME size. The
  workflow does this already, so normally it just works.
- If you resize the image, or change `det_size` or `order`, RE-READ the grid.
  The old numbers may now point at different faces.
- Small images lose faces. 187x266 found only 15 of 19. Upscale first.

### det_size
`det_size = 0` means AUTO: `clamp(round(max(w,h)/2 /32)*32, 640, 1280)`.
Raise it manually if small faces are missed; lower it for speed.

Bigger is not always better - 1024 on the 1125x1599 image found 18 faces where
auto's 800 found 20. If faces are missing, try a few values.

### Bug worked around
ReActor seeds `ANALYSIS_MODELS` with only the keys "320" and "640"
(reactor_swapper.py:69) and then does a bare `ANALYSIS_MODELS[str(det_size[0])]`
lookup, so ANY other det_size raised `KeyError: '1024'`. `_allow_det_size()` in
comfyui-face-tools pre-seeds the key so arbitrary sizes work. ReActor itself is
untouched, so a ReActor update will not clobber the fix.

### Face resolution vs quality
`inswapper_128` outputs 128x128 per face. A face smaller than ~128 px in the
target gets upscaled and looks soft; much larger than 128 px gets downscaled
and looks soft too. Sharpest results come from faces near 128-256 px.
On the 375x533 poster the faces are ~35 px, which is why that output is mushy -
upscale 3x first and the same swap looks far better.
