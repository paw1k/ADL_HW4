import json
from pathlib import Path

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

def _center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2

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
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 5
) -> list[dict]:
    """
    Extract kart objects from the info.json file, including their center points and identify the center kart.
    Filters out karts that are out of sight (outside the image boundaries).

    Args:
        info_path: Path to the corresponding info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 100)
        img_height: Height of the image (default: 150)

    Returns:
        List of kart objects, each containing:
        - instance_id: The track ID of the kart
        - kart_name: The name of the kart
        - center: (x, y) coordinates of the kart's center
        - is_center_kart: Boolean indicating if this is the kart closest to image center
    """

    # raise NotImplementedError("Not implemented")

    with open(info_path) as f:
        info = json.load(f)

    detections = info["detections"][view_index]
    name_table = info.get("names", {})          # ← NEW: graceful fallback
    get_name   = lambda tid: name_table.get(str(tid), f"kart_{tid}")  # ← NEW

    ego_id  = info.get("ego_id")                # sometimes absent in old files

    # scale factors for 600 × 400 → target-size
    sx = img_width  / ORIGINAL_WIDTH
    sy = img_height / ORIGINAL_HEIGHT

    karts = []
    for cls, track_id, x1, y1, x2, y2 in detections:
        if cls != 1:                                   # only karts (class id = 1)
            continue
        # … clipping & size check unchanged …

        cx, cy = _center(box)

        karts.append(
            dict(
                instance_id=track_id,
                kart_name=get_name(track_id),  # ← use helper
                center=(cx, cy),
                is_center_kart=False,
                is_left=None,
                is_front=None,
            )
        )

    # determine the ego (closest to image centre)
    img_cx, img_cy = img_width / 2, img_height / 2
    if not karts:
        return []

    ego_idx = min(range(len(karts)), key=lambda i: (karts[i]["instance_id"] != ego_id,  # prefer labelled ego_id
                                                    (karts[i]["center"][0]-img_cx) ** 2 +
                                                    (karts[i]["center"][1]-img_cy) ** 2))
    for i, k in enumerate(karts):
        k["is_center_kart"] = i == ego_idx
        k["is_left"]  = k["center"][0] < img_cx
        k["is_front"] = k["center"][1] < img_cy  # smaller y ⇒ higher in image ⇒ in front

    return karts


def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.

    Args:
        info_path: Path to the info.json file

    Returns:
        Track name as a string
    """

    # raise NotImplementedError("Not implemented")
    with open(info_path) as f:
        info = json.load(f)
    return info["track"]



def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list[dict]:
    """
    Generate question-answer pairs for a given view.

    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 100)
        img_height: Height of the image (default: 150)

    Returns:
        List of dictionaries, each containing a question and answer
    """
    # 1. Ego car question
    # What kart is the ego car?

    # 2. Total karts question
    # How many karts are there in the scenario?

    # 3. Track information questions
    # What track is this?

    # 4. Relative position questions for each kart
    # Is {kart_name} to the left or right of the ego car?
    # Is {kart_name} in front of or behind the ego car?

    # 5. Counting questions
    # How many karts are to the left of the ego car?
    # How many karts are to the right of the ego car?
    # How many karts are in front of the ego car?
    # How many karts are behind the ego car?

    # raise NotImplementedError("Not implemented")
    qa_pairs: list[dict] = []

    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    if not karts:
        return qa_pairs

    track = extract_track_info(info_path)

    ego      = next(k for k in karts if k["is_center_kart"])
    others   = [k for k in karts if not k["is_center_kart"]]

    # 1. Ego-kart identity
    qa_pairs.append(
        dict(question="What kart is the ego car?", answer=ego["kart_name"])
    )

    # 2. Total count
    qa_pairs.append(
        dict(question="How many karts are there in the scenario?", answer=str(len(karts)))
    )

    # 3. Track name
    qa_pairs.append(dict(question="What track is this?", answer=track))

    # 4. Relative positions
    for k in others:
        lr = "left" if k["is_left"] else "right"
        fb = "front" if k["is_front"] else "back"

        qa_pairs.append(
            dict(
                question=f"Is {k['kart_name']} to the left or right of the ego car?",
                answer=lr,
            )
        )
        qa_pairs.append(
            dict(
                question=f"Is {k['kart_name']} in front of or behind the ego car?",
                answer=fb,
            )
        )
        qa_pairs.append(
            dict(
                question=f"Where is {k['kart_name']} relative to the ego car?",
                answer=f"{fb} and {lr}",
            )
        )

    # 5. Counting by region
    left_cnt  = sum(k["is_left"]  for k in others)
    right_cnt = sum(not k["is_left"] for k in others)
    front_cnt = sum(k["is_front"] for k in others)
    back_cnt  = sum(not k["is_front"] for k in others)

    qa_pairs.extend(
        [
            dict(question="How many karts are to the left of the ego car?",  answer=str(left_cnt)),
            dict(question="How many karts are to the right of the ego car?", answer=str(right_cnt)),
            dict(question="How many karts are in front of the ego car?",    answer=str(front_cnt)),
            dict(question="How many karts are behind the ego car?",         answer=str(back_cnt)),
        ]
    )

    return qa_pairs



def check_qa_pairs(info_file: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze
    """
    # Find corresponding image file
    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    # Visualize detections
    annotated_image = draw_detections(str(image_file), info_file)

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

    # Generate QA pairs
    qa_pairs = generate_qa_pairs(info_file, view_index)

    # Print QA pairs
    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 50)


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_qa.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_qa_pairs})


if __name__ == "__main__":
    main()
