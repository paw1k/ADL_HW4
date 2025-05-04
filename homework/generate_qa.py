import json
from pathlib import Path
from typing import Tuple

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# Define object type mapping
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

# Define colors for different object types (RGB format)
COLORS = {
    1: (0, 255, 0),  # Green for karts
    2: (255, 0, 0),  # Blue for track boundaries
    3: (0, 0, 255),  # Red for track elements
    4: (255, 255, 0),  # Cyan for special elements
    5: (255, 0, 255),  # Magenta for special elements
    6: (0, 255, 255),  # Yellow for special elements
}

# Original image dimensions for the bounding box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def _clip_bbox(box, w, h):
    x1, y1, x2, y2 = box
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)

def _center(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """Return (cx, cy) for a (x1, y1, x2, y2) box."""
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


# -------------------------------------------------
def _relative_image_path(info_path: Path, view_index: int) -> str | None:
    """
    Return the frame filename *relative to the data dir* so
    VQADataset can join it with its own data_dir later on.
    If the frame isn't found, return None.
    """
    base_name = info_path.stem.replace("_info", "")
    try:
        img_path = _find_image_for_view(info_path, base_name, view_index)
        # example:  data/train/00000_00_im.jpg  ->  train/00000_00_im.jpg
        return str(img_path.relative_to(info_path.parents[1]))
    except FileNotFoundError:
        return None


def _find_image_for_view(info_path: Path, base: str, view_index: int) -> Path:
    """
    Locate the rendered frame that corresponds to   base + f"_{view:02d}_im.(jpg|png)".
    Returns a pathlib.Path or raises FileNotFoundError if nothing matches.
    """
    patterns = [
        f"{base}_{view_index:02d}_im.jpg",
        f"{base}_{view_index:02d}_im.png",
    ]
    for pat in patterns:
        hits = list(info_path.parent.glob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"No frame image found for view {view_index} next to {info_path.name} "
        f"(looked for *.jpg and *.png)."
    )



def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (frame_id, view_index)
    """
    filename = Path(image_path).name
    # Format is typically: XXXXX_YY_im.png where XXXXX is frame_id and YY is view_index
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)  # Convert hex to decimal
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0  # Default values if parsing fails


def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on the image.

    Args:
        image_path: Path to the image file
        info_path: Path to the corresponding info.json file
        font_scale: Scale of the font for labels
        thickness: Thickness of the bounding box lines
        min_box_size: Minimum size for bounding boxes to be drawn

    Returns:
        The annotated image as a numpy array
    """
    # Read the image using PIL
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Get image dimensions
    img_width, img_height = pil_image.size

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Extract frame ID and view index from image filename
    _, view_index = extract_frame_info(image_path)

    # Get the correct detection frame based on view index
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return np.array(pil_image)

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # Draw each detection
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Get color for this object type
        if track_id == 0:
            color = (255, 0, 0)
        else:
            color = COLORS.get(class_id, (255, 255, 255))

        # Draw bounding box using PIL
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=thickness)

    # Convert PIL image to numpy array for matplotlib
    return np.array(pil_image)


def extract_kart_objects(
    info_path: str,
    view_index: int,
    img_width: int = 150,
    img_height: int = 100,
    min_box_size: int = 5,
) -> list[dict]:
    """
    Parse *info.json* and return a normalised list describing every visible kart.

    Each element::
        {
          "instance_id":  <track‑id>,
          "kart_name":    <string>,           # if names table present
          "center":       (cx, cy),           # after scaling to (img_width×img_height)
          "is_center_kart": bool,             # ego (closest to img centre or labelled)
          "is_left":      bool,               # relative to img centre
          "is_front":     bool,               # y smaller  ⇒ in front
        }
    """
    with open(info_path, "r") as f:
        info = json.load(f)

    detections = info["detections"][view_index]
    names      = {str(k): v for k, v in info.get("names", {}).items()}
    ego_id     = info.get("ego_id")            

    sx = img_width  / ORIGINAL_WIDTH
    sy = img_height / ORIGINAL_HEIGHT

    karts = []
    for cls, track_id, x1, y1, x2, y2 in detections:
        if int(cls) != 1:
            continue                  
        box_scaled = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
        x1s, y1s, x2s, y2s = box_scaled
        w, h = x2s - x1s, y2s - y1s
        if w < min_box_size or h < min_box_size:
            continue
        if x2s < 0 or x1s > img_width or y2s < 0 or y1s > img_height:
            continue

        cx, cy = _center(box_scaled)
        karts.append(
            dict(
                instance_id=int(track_id),
                kart_name=names.get(str(track_id), f"kart_{track_id}"),
                center=(cx, cy),
                is_center_kart=False,      
                is_left=None,
                is_front=None,
            )
        )

    if not karts:
        return []

    img_cx, img_cy = img_width / 2.0, img_height / 2.0

    def _dist_sq(k):
        dx, dy = k["center"][0] - img_cx, k["center"][1] - img_cy
        return dx * dx + dy * dy

    ego_idx = 0
    if ego_id is not None:
        for i, k in enumerate(karts):
            if k["instance_id"] == ego_id:
                ego_idx = i
                break
        else:
            ego_idx = min(range(len(karts)), key=lambda i: _dist_sq(karts[i]))
    else:
        ego_idx = min(range(len(karts)), key=lambda i: _dist_sq(karts[i]))

    for i, k in enumerate(karts):
        k["is_center_kart"] = i == ego_idx
        k["is_left"]  = k["center"][0] < img_cx
        k["is_front"] = k["center"][1] < img_cy  

    return karts


def extract_track_info(info_path: str) -> str:
    """
    Returns the lowercase track name recorded in *info.json*.
    Falls back to `"unknown"` if the key is missing.
    """
    with open(info_path, "r") as f:
        info = json.load(f)
    return str(info.get("track", "unknown")).lower()


# --------------------------------------------------------------------
def generate_qa_pairs(
    info_path: str,
    view_index: int,
    img_width: int = 150,
    img_height: int = 100,
) -> list[dict]:
    """
    Build the 5 families of QA pairs for a single `*_info.json` frame.

    Every dictionary now contains **image_file** so the finetuning
    pipeline can locate the picture later:

        {
            "image_file": "train/00000_03_im.jpg",
            "question"  : "...",
            "answer"    : "..."
        }

    If the frame image is missing, image_file is set to None.
    """
    info_path = Path(info_path)
    rel_image = _relative_image_path(info_path, view_index)

    qa_pairs: list[dict] = []

    # -----------------------------------------------------------------
    #  Parse detections & meta data
    # -----------------------------------------------------------------
    karts = extract_kart_objects(str(info_path), view_index, img_width, img_height)
    if not karts:                        # nothing detected ⇒ nothing to ask
        return qa_pairs

    track = extract_track_info(str(info_path))

    ego    = next(k for k in karts if k["is_center_kart"])
    others = [k for k in karts if not k["is_center_kart"]]

    # -----------------------------------------------------------------
    #  1. Ego‑kart identity
    # -----------------------------------------------------------------
    qa_pairs.append(
        dict(
            image_file=rel_image,
            question="What kart is the ego car?",
            answer=ego["kart_name"],
        )
    )

    # -----------------------------------------------------------------
    #  2. Total kart count
    # -----------------------------------------------------------------
    qa_pairs.append(
        dict(
            image_file=rel_image,
            question="How many karts are there in the scenario?",
            answer=str(len(karts)),
        )
    )

    # -----------------------------------------------------------------
    #  3. Track
    # -----------------------------------------------------------------
    qa_pairs.append(
        dict(
            image_file=rel_image,
            question="What track is this?",
            answer=track,
        )
    )

    # -----------------------------------------------------------------
    #  4. Relative position questions
    # -----------------------------------------------------------------
    for k in others:
        lr = "left" if k["is_left"] else "right"
        fb = "front" if k["is_front"] else "back"

        qa_pairs.extend(
            [
                dict(
                    image_file=rel_image,
                    question=f"Is {k['kart_name']} to the left or right of the ego car?",
                    answer=lr,
                ),
                dict(
                    image_file=rel_image,
                    question=f"Is {k['kart_name']} in front of or behind the ego car?",
                    answer=fb,
                ),
                dict(
                    image_file=rel_image,
                    question=f"Where is {k['kart_name']} relative to the ego car?",
                    answer=f"{fb} and {lr}",
                ),
            ]
        )

    # -----------------------------------------------------------------
    #  5. Region counts
    # -----------------------------------------------------------------
    left_cnt  = sum(k["is_left"] for k in others)
    right_cnt = len(others) - left_cnt
    front_cnt = sum(k["is_front"] for k in others)
    back_cnt  = len(others) - front_cnt

    qa_pairs.extend(
        [
            dict(
                image_file=rel_image,
                question="How many karts are to the left of the ego car?",
                answer=str(left_cnt),
            ),
            dict(
                image_file=rel_image,
                question="How many karts are to the right of the ego car?",
                answer=str(right_cnt),
            ),
            dict(
                image_file=rel_image,
                question="How many karts are in front of the ego car?",
                answer=str(front_cnt),
            ),
            dict(
                image_file=rel_image,
                question="How many karts are behind the ego car?",
                answer=str(back_cnt),
            ),
        ]
    )

    return qa_pairs



def check_qa_pairs(info_file: str, view_index: int = 0):
    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")

    image_file: Path | None = None
    try:
        image_file = _find_image_for_view(info_path, base_name, view_index)
    except FileNotFoundError as e:
        print(e)
        print("Skipping visualisation; will still print generated QA pairs.\n")

    qa_pairs = generate_qa_pairs(info_file, view_index)
    if not qa_pairs:
        print("No karts detected – no QA pairs generated.")
        return

    print("\nQuestion–Answer pairs")
    print("-" * 60)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 60)

    if image_file is not None:
        annotated = draw_detections(str(image_file), info_file)

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(annotated)
        ax.axis("off")

        frame_id, _ = extract_frame_info(str(image_file))
        ax.set_title(
            f"{info_path.name}  |  frame {frame_id}  |  view {view_index}",
            fontsize=14,
        )
        plt.show()
    else:
        print("(No frame image found – visualisation skipped.)")


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_qa.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


# -------------------------------------------------
#  ADD THIS NEAR THE BOTTOM OF generate_qa.py
# -------------------------------------------------
def generate_dataset(
    data_dir: str,
    output_dir: str,
    split: str = "train",
    img_width: int = 150,
    img_height: int = 100,
):
    """
    Create a single  *_qa_pairs.json  file for an entire split.

    Example
    -------
    python -m homework.generate_qa generate \
           --data_dir ./data --output_dir ./data --split train
    """
    import glob, os, json
    from tqdm import tqdm

    info_files = sorted(glob.glob(os.path.join(data_dir, split, "*_info.json")))
    all_pairs  = []

    for info in tqdm(info_files, desc=f"processing {split} split"):
        # every info file has 10 camera views
        for view in range(10):
            all_pairs.extend(
                generate_qa_pairs(info, view, img_width=img_width, img_height=img_height)
            )

    outfile = os.path.join(output_dir, f"{split}_qa_pairs.json")
    with open(outfile, "w") as f:
        json.dump(all_pairs, f, indent=2)

    print(f"wrote {len(all_pairs):,} QA pairs → {outfile}")


def main():
    fire.Fire(
        {
            "check":    check_qa_pairs,
            "generate": generate_dataset,   # ← NEW
        }
    )


if __name__ == "__main__":
    main()
