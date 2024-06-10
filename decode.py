import tensorflow as tf
from typing import Any, Dict
import io
import os
import numpy as np
import skia
import tensorflow as tf
from typing import Any, Dict
from PIL import Image
from matplotlib import pyplot as plt
import cv2
from tqdm import tqdm
import numpy as np

def get_type_string(val: str) -> str:
    if val=="imageElement" or val=="maskElement":
        return "img", 0
    elif val=="svgElement":
        return "svg", 1
    elif val=="coloredBackground":
        return "bg", 2
    elif val=="textElement":
        return "txt", 3
    else:
        raise Exception

def parse(serialized: bytes) -> Dict[str, tf.Tensor]:
    """Explicitly parse specified fields."""
    context, sequence, _ = tf.io.parse_sequence_example(
        serialized,
        context_features={
            "id": tf.io.FixedLenFeature((), tf.string),
            "group": tf.io.FixedLenFeature((), tf.string),
            "format": tf.io.FixedLenFeature((), tf.string),
            "category": tf.io.FixedLenFeature((), tf.string),
            "canvas_height": tf.io.FixedLenFeature((), tf.int64),
            "canvas_width": tf.io.FixedLenFeature((), tf.int64),
            "length": tf.io.FixedLenFeature((), tf.int64),
        },
        sequence_features={
            "type": tf.io.FixedLenSequenceFeature((), tf.string),
            "left": tf.io.FixedLenSequenceFeature((), tf.float32),
            "top": tf.io.FixedLenSequenceFeature((), tf.float32),
            "width": tf.io.FixedLenSequenceFeature((), tf.float32),
            "height": tf.io.FixedLenSequenceFeature((), tf.float32),
            "color": tf.io.FixedLenSequenceFeature((3,), tf.int64),
            "opacity": tf.io.FixedLenSequenceFeature((), tf.float32),
            "image_bytes": tf.io.FixedLenSequenceFeature((), tf.string),
            "text": tf.io.FixedLenSequenceFeature((), tf.string),
            "font": tf.io.FixedLenSequenceFeature((), tf.string),
            "font_size": tf.io.FixedLenSequenceFeature((), tf.float32),
            "text_align": tf.io.FixedLenSequenceFeature((), tf.string),
            "angle": tf.io.FixedLenSequenceFeature((), tf.float32),
        })
    return {**context, **sequence}

def get_bbox(map):
    map = map!=0
    col_has_one = map.any(axis=0) 
    row_has_one = map.any(axis=1)
    y1, x1 = map.shape
    y2, x2 = 0, 0
    for idx,has_one in enumerate(col_has_one.tolist()):
        if has_one:
            x1 = idx
            break
    for idx,has_one in enumerate(col_has_one.tolist()):
        if has_one:
            x2 = idx
    for idx,has_one in enumerate(row_has_one.tolist()):
        if has_one:
            y1 = idx
            break
    for idx,has_one in enumerate(row_has_one.tolist()):
        if has_one:
            y2 = idx
    return (y1, x1), (y2, x2)

def render(example: Dict[str, tf.Tensor], save_path: str="", max_size: float=1024.) -> bytes:
    global elementTypes
    """Render parsed sequence example onto an image and return as PNG bytes."""
    canvas_width = example["canvas_width"]
    canvas_height = example["canvas_height"]

    scale = min(1.0, max_size / canvas_width, max_size / canvas_height)
    surface = skia.Surface(int(scale * canvas_width), int(scale * canvas_height))
    maps = []
    with surface as canvas:
        # canvas.scale(scale, scale)
        final_canvas_height = int(scale * canvas_height)
        final_canvas_width = int(scale * canvas_width)
        for index in range(example["length"]):
            if example["type"][index].decode()=="textElement":
                continue
            with io.BytesIO(example["image_bytes"][index]) as f:
                image = skia.Image.open(f)
            left = int(example["left"][index] * canvas_width * scale)
            top = int(example["top"][index] * canvas_height * scale)
            width = int(example["width"][index] * canvas_width * scale)
            height = int(example["height"][index] * canvas_height * scale)
            # image.save(save_path + f"_{index}_" + example["type"][index].decode() + ".jpg")

            width = width if width!=0 else 1
            height = height if height!=0 else 1

            resized_image = image.resize(width, height)
            resized_image_alpha = resized_image.toarray()[..., -1]
            resized_image_alpha[resized_image_alpha<128] = 0
            resized_image_alpha[resized_image_alpha>127] = 1

            where_x_start = max(0, left) 
            where_x_end = min(final_canvas_width, left+width)
            where_y_start = max(0, top)
            where_y_end = min(final_canvas_height, top+height) 

            what_x_start = max(0, left) - left
            what_x_end = min(final_canvas_width, left+width) - left
            what_y_start = max(0, top) - top
            what_y_end = min(final_canvas_height, top+height) - top

            for map,_ in maps:
                try:
                    map[where_y_start:where_y_end, where_x_start:where_x_end] = map[where_y_start:where_y_end, where_x_start:where_x_end] * (1 - resized_image_alpha[what_y_start:what_y_end, what_x_start:what_x_end])
                except:
                    print(save_path + f"_{index}_" + example["type"][index].decode() + ".jpg")
                    image.save(save_path + f"_{index}_" + example["type"][index].decode() + ".jpg", skia.kJPEG)
                    print(scale)
                    print(canvas_width, canvas_height)
                    print(example["left"][index], example["top"][index], example["width"][index], example["height"][index])
                    print(0, 0, final_canvas_width, final_canvas_height)
                    print(left, top, width, height)
                    print(where_x_start, where_y_start, where_x_end, where_y_end)
                    print(what_x_start, what_y_start, what_x_end, what_y_end)
                    print("\n\n")
            if example["type"][index].decode() in ["imageElement", "maskElement", "svgElement", "coloredBackground"]:
                t = example["type"][index].decode()
                new_map = np.zeros((final_canvas_height, final_canvas_width), dtype=np.uint8)
                try:
                    new_map[where_y_start:where_y_end, where_x_start:where_x_end] = resized_image_alpha[what_y_start:what_y_end, what_x_start:what_x_end]
                    maps.append((new_map, t))
                except:
                    print(t)
                    print(save_path + f"_{index}_" + example["type"][index].decode() + ".jpg")
                    image.save(save_path + f"_{index}_" + example["type"][index].decode() + ".jpg", skia.kJPEG)
                    print(scale)
                    print(canvas_width, canvas_height)
                    print(example["left"][index], example["top"][index], example["width"][index], example["height"][index])
                    print(0, 0, final_canvas_width, final_canvas_height)
                    print(left, top, width, height)
                    print(where_x_start, where_y_start, where_x_end, where_y_end)
                    print(what_x_start, what_y_start, what_x_end, what_y_end)
                    print("\n\n")

            # elif example["type"][index].decode()=="svgElement":
            #     l = max(0, left) / final_canvas_width
            #     t = max(0, top) / final_canvas_height
            #     w = min(final_canvas_width, width) / final_canvas_width
            #     h = min(final_canvas_height, height) / final_canvas_height
            #     images_bbox.append((1, l, t, w, h))
            rect = skia.Rect.MakeXYWH(left, top, width, height)

            # angle = example["angle"][index]
            # if angle != 0:
            #     degree = 180. * angle / np.pi
            #     canvas.save()
            #     canvas.rotate(degree, left + width / 2., top + height / 2.)

            canvas.drawImageRect(image, rect)
            # if angle != 0:
            #     canvas.restore()

    image = surface.makeImageSnapshot()
    bboxes = []
    save_maps = []
    for map,tp in maps:
        (y1, x1), (y2, x2) = get_bbox(map)
        center_x = (x2 + x1) / 2 / final_canvas_width
        center_y = (y2 + y1) / 2 / final_canvas_height
        width = (x2 - x1) / final_canvas_width
        height = (y2 - y1) / final_canvas_height
        tp_str,tp_num = get_type_string(tp)
        bboxes.append((tp_num, center_x, center_y, width, height))
        save_maps.append((Image.fromarray((map * 255).astype(np.uint8)), tp_str))
    with io.BytesIO() as f:
        image.save(f, skia.kPNG)
        return f.getvalue(), save_maps, bboxes


types = ["val", "train", "test"]
for tp in types:
    print(tp)
    dataset = tf.data.Dataset.list_files(f"{tp}-*.tfrecord")
    dataset = tf.data.TFRecordDataset(dataset)
    dataset = dataset.map(parse)

    images_output_folder_path = os.path.join("new_dataset", "images", tp)
    error_output_folder_path = os.path.join("new_dataset", "error", tp)
    maps_output_folder_path = os.path.join("new_dataset", "maps", tp)
    labels_output_folder_path = os.path.join("new_dataset", "labels", tp)

    os.makedirs(images_output_folder_path, exist_ok=True)
    os.makedirs(error_output_folder_path, exist_ok=True)
    os.makedirs(maps_output_folder_path, exist_ok=True)
    os.makedirs(labels_output_folder_path, exist_ok=True)

    ## dataset.take(20).as_numpy_iterator()
    for idx,example in tqdm(enumerate(dataset.as_numpy_iterator())):
        file_name = f"design_{str(idx+1).rjust(6, '0')}"

        image, maps, bboxes = render(example, save_path=os.path.join(error_output_folder_path, file_name))

        image_path = os.path.join(images_output_folder_path, f"{file_name}.jpg")
        with open(image_path, "wb") as fh:
            fh.write(image)

        label_path = os.path.join(labels_output_folder_path, f"{file_name}.txt")
        label_fh = open(label_path, "w")
        for idx,(map,annotation) in enumerate(zip(maps, bboxes)):
            map, tp_str = map
            map_path = os.path.join(maps_output_folder_path, f"{file_name}_{str(idx+1).rjust(3, '0')}_{tp_str}.jpg")
            map.save(map_path)
            for x in annotation:
                label_fh.write(str(x))
                label_fh.write(" ")
            label_fh.write("\n")
        label_fh.close()
    
