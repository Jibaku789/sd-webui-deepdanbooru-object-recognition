"""
    @author: Jibaku789
    @version: 1.2.1
    @date: December 2023
"""

import modules.scripts as scripts
import gradio as gr
import os
import time
import torch
import numpy as np
import copy
import json
import random
import uuid
from PIL import ImageDraw, Image, ImageFont

import matplotlib.pylab as plt

from modules import script_callbacks
from modules import devices, images
from modules.deepbooru import DeepDanbooru
from modules import shared

class DeepDanbooruWrapper:

    def __init__(
        self,
    ):

        self.dd_classifier = DeepDanbooru()
        self.cache = {}
        self.enable_cache = True

    def start(self):
        print("Starting DeepDanboru")
        self.dd_classifier.start()

    def stop(self):
        print("Stopping DeepDanboru")
        self.dd_classifier.stop()

    def evaluate_model(self, pil_image, image_id="", minimal_threshold=0):

        if self.enable_cache:
            if image_id and image_id in self.cache:
                return self.cache[image_id]

        # Input image should be 512x512 before reach this point

        pic = images.resize_image(0, pil_image.convert("RGB"), 512, 512)

        #pic = pil_image.convert("RGB")
        a = np.expand_dims(np.array(pic, dtype=np.float32), 0) / 255

        with torch.no_grad(), devices.autocast():
            x = torch.from_numpy(a).to(devices.device)
            y = self.dd_classifier.model(x)[0].detach().cpu().numpy()

        probability_dict = {}
        for tag, probability in zip(self.dd_classifier.model.tags, y):

            if probability < minimal_threshold:
                continue

            if tag.startswith("rating:"):
                continue

            probability_dict[tag] = probability

        if self.enable_cache:
            self.cache[image_id] = probability_dict

        return probability_dict

class DeepDanbooruObjectDrawer:

    def __init__(
        self,
        pil_image,
        title,
        export_directory
    ):

        w, h = pil_image.size
        self.original_pil_image = self.resize(pil_image, max(w, h))
        self.rect_pil_image = self.original_pil_image.copy()
        self.pil_image = self.resize(pil_image, 512)
        self.title = title
        self.export_directory = export_directory

    def resize(self, pil_image, to_scale):

        target_size = (to_scale, to_scale)
        background = Image.new('RGB', target_size)

        width, height = pil_image.size
        new_w, new_h = 0, 0
        x1, y1 = 0, 0

        if width > height:
            new_w = to_scale
            new_h = int(to_scale * (height/width))
            x1 = 0
            y1 = int((to_scale - new_h)/2)
            
        else:
            new_h = to_scale
            new_w = int(to_scale * (width/height))
            x1 = int((to_scale - new_w)/2)
            y1 = 0

        newsize = (new_w, new_h)
        newImg1 = pil_image.resize(newsize)
        background.paste(newImg1, (x1, y1))
        return background

    def crop(self, top, left, bottom, right, export=False, image_to_use="NORM"):
        # All values shoud be between 0-512

        size = 512
        if image_to_use == "ORIG":
            size = self.original_pil_image.size[0]
        elif image_to_use == "RECT":
            size = self.rect_pil_image.size[0]

        y1 = int((top/512) * size)
        x1 = int((left/512) * size)
        y2 = int((bottom/512) * size)
        x2 = int((right/512) * size)

        target_size = (size, size)

        if image_to_use == "ORIG":
            im1 = self.original_pil_image.resize(target_size)
        elif image_to_use == "RECT":
            im1 = self.rect_pil_image.resize(target_size)
        else:
            im1 = self.pil_image.resize(target_size)

        im2 = im1.crop((x1, y1, x2, y2))

        background = Image.new('RGB', target_size)
        background.paste(im2, (x1, y1))

        if export:
            prefix = f"{self.export_directory}/{self.title}_{top}_{left}_{bottom}_{right}"
            background.save(f"{prefix}.png")

        return background


    def draw_rect(self, borders, title):
        # Borders coordinates should be between 0-512

        im1 = self.rect_pil_image
        draw = ImageDraw.Draw(im1)

        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
            0
        )

        font_path = os.path.abspath(os.path.join(__file__, "..", "..", "resources", "Arial.ttf"))
        line_width = int((im1.size[0]/512) * 1.5)
        text_width = int((im1.size[0]/512) * 10)
        font = ImageFont.truetype(font_path, text_width)

        top = int((borders["top"]/512) * im1.size[0])
        left = int((borders["left"]/512) * im1.size[0])
        bottom = int((borders["bottom"]/512) * im1.size[0])
        right = int((borders["right"]/512) * im1.size[0])

        draw.rectangle(
            [
                (left, top), 
                (right, bottom)
            ],
            outline=color,
            width=line_width
        )

        draw.text(
            (left + line_width + 1, top + line_width + 1),
            title,
            fill=color,
            font=font
        )

        self.rect_pil_image = im1

class DeepDanbooruObjectRecognitionNode:

    def __init__(
        self,
        dd_wrapper,
        pil_image,
        tag,
        export_directory
    ):
        self.dd_wrapper = dd_wrapper
        self.tag = tag

        self.export_directory = export_directory

        self.drawer = DeepDanbooruObjectDrawer(
            pil_image,
            tag,
            self.export_directory
        )

        self.pil_image = self.drawer.pil_image.copy()


    def create_heatmaps(self, kernel_size_x, kernel_size_y, step_x, step_y, minimal_percentage):

        # Headmap approach
        current_y = 0
        dots = []
        bests = []

        while current_y + kernel_size_y < 512:

            current_x = 0
            x_dots = []

            while current_x + kernel_size_x < 512:

                im1 = self.drawer.crop(
                    current_y,
                    current_x,
                    current_y + kernel_size_y,
                    current_x + kernel_size_x,
                    #export=True
                )

                iid = f'{current_y}-{current_x}-{current_y+kernel_size_y}-{current_x+kernel_size_x}'
                prob = self.dd_wrapper.evaluate_model(im1, iid)

                if self.tag in prob:
                    prob = prob[self.tag]
                else:
                    prob = 0

                if prob > minimal_percentage:
                    bests.append(
                        {
                            "top": current_y,
                            "left": current_x,
                            "bottom": current_y + kernel_size_y,
                            "right": current_x + kernel_size_x,
                            "prob": float(prob)
                        }
                    )

                print(f"{iid}, {prob}")
                x_dots.append(prob)

                current_x += step_x

            dots.insert(0, x_dots)
            current_y += step_y

        with open(f"{self.export_directory}/dots_{self.tag}.json", "w") as _f:
            _f.write(json.dumps(bests, indent=4))

        dots = np.array(dots)

        # Create heatmap
        fig, ax = plt.subplots()
        c = ax.pcolormesh(dots, cmap='gray', vmin=0, vmax=1)

        fig.colorbar(c, ax=None)
        fig.canvas.draw()
        
        image_name = f"{self.export_directory}/heatmap_{self.tag}.png"
        fig.savefig(image_name, bbox_inches='tight', pad_inches=0)

        # Function to delete duplicates entries
        def delete_duplicated(bests, axis="X"):

            def merge_pair(iterable_bounces, axis, i, j):

                border = iterable_bounces[i]
                other_border = iterable_bounces[j]
                new_border = {}

                new_border = {
                    "top": min(border["top"], other_border["top"]),
                    "left": min(border["left"], other_border["left"]),
                    "bottom": max(border["bottom"], other_border["bottom"]),
                    "right": max(border["right"], other_border["right"]),
                    "prob": max(border["prob"], other_border["prob"])
                }

                return new_border

            pairs = ["x"]
            iterable_bounces = bests
            while pairs:

                pairs = []

                for i in range(0, len(iterable_bounces)):
                    for j in range(i+1, len(iterable_bounces)):

                        border = iterable_bounces[i]
                        other_border = iterable_bounces[j]

                        if axis == "X":

                            # Border inside the limits of the new one
                            if other_border["left"] > border["left"] and \
                               other_border["left"] < border["right"]:

                                if other_border["top"] > border["top"]  and \
                                   other_border["top"] < border["bottom"]:
                                    continue

                                if i+1 == j:
                                    my_pair = list(sorted([i, j]))
                                    if my_pair not in pairs:
                                        pairs.append(my_pair) 
                        else:

                            # Border inside the limits of the new one
                            if other_border["top"] > border["top"]  and \
                               other_border["top"] < border["bottom"]:

                                if other_border["left"] > border["left"] and \
                                   other_border["left"] < border["right"]:
                                    continue

                                if i+1 == j:
                                    my_pair = list(sorted([i, j]))
                                    if my_pair not in pairs:
                                        pairs.append(my_pair) 

                # Merge the pairs
                #print(json.dumps(iterable_bounces, indent=4))
                #print(pairs)
                if pairs:

                    new_iterable = []
                    current_pair = []

                    for pair in pairs:

                        if not current_pair:
                            current_pair = [pair[0], pair[1]]
                            continue

                        if pair[0] == current_pair[1]:
                            current_pair[1] = pair[1]
                            continue

                        new_border = merge_pair(iterable_bounces, axis, current_pair[0], current_pair[1])
                        new_iterable.append(new_border)
                        current_pair = [pair[0], pair[1]]

                    if current_pair:
                        new_border = merge_pair(iterable_bounces, axis, current_pair[0], current_pair[1])
                        new_iterable.append(new_border)

                    # Add missing number
                    for i in range(0, len(iterable_bounces)):

                        found = False
                        for pair in pairs:
                            if i in pair:
                                found = True
                                break

                        if not found:
                            new_iterable.append(iterable_bounces[i])

                    # Sort new list
                    if axis == "X":
                        new_iterable = sorted(
                            new_iterable, key = lambda x: x["left"]
                        )
                    else:
                        new_iterable = sorted(
                            new_iterable, key = lambda x: x["top"]
                        )

                    iterable_bounces = new_iterable

            return iterable_bounces

        bests = delete_duplicated(bests, axis="X")
        bests = delete_duplicated(bests, axis="Y")

        c = 0
        for best in bests:
            c += 1
            self.drawer.title = f"Best_{c}_{self.tag}"
            self.drawer.crop(
                best["top"],
                best["left"],
                best["bottom"],
                best["right"],
                export=True,
                image_to_use="ORIG"
            )

        print(bests)
        return bests


    def rect_tag(self, steps = 10, subdivisions = 3, tolerance = 0.05):

        # Init prob
        print(f"Evaluating: {self.tag}")

        im0 = self.drawer.crop(0, 0, 512, 512)
        initial_prob = self.dd_wrapper.evaluate_model(im0, "0-0-512-512")

        if self.tag not in initial_prob:
            return None

        initial_prob = initial_prob[self.tag]

        current_borders = {
            "top": 0,
            "left": 0,
            "bottom": 512,
            "right": 512
        }

        best_prob = initial_prob
        best_status = copy.deepcopy(current_borders)

        for s in range(0, steps):

            # Calculate proportions
            changed = False

            x_diff = current_borders["right"] - current_borders["left"]
            y_diff = current_borders["bottom"] - current_borders["top"]

            propotions = {
                "top": int(current_borders["top"] + (y_diff * ((subdivisions-1)/subdivisions))),
                "left": int(current_borders["left"] + (x_diff * ((subdivisions-1)/subdivisions))),
                "bottom": int(current_borders["bottom"] - (y_diff * ((subdivisions-1)/subdivisions))),
                "right": int(current_borders["right"] - (x_diff * ((subdivisions-1)/subdivisions)))
            }

            print(current_borders)

            # Calculate the new four sectors
            borders = []

            borders.append({
                "top": current_borders["top"],
                "left": current_borders["left"],
                "bottom": propotions["top"],
                "right": propotions["left"]
            })

            borders.append({
                "top": current_borders["top"],
                "left": propotions["right"],
                "bottom": propotions["top"],
                "right": current_borders["right"]
            })

            borders.append({
                "top": propotions["bottom"],
                "left": current_borders["left"],
                "bottom": current_borders["bottom"],
                "right": propotions["left"]
            })

            borders.append({
                "top": propotions["bottom"],
                "left": propotions["right"],
                "bottom": current_borders["bottom"],
                "right": current_borders["right"]
            })

            # Evaluate the new regions and determine best
            for border in borders:

                im1 = self.drawer.crop(
                    border["top"],
                    border["left"],
                    border["bottom"],
                    border["right"]
                )

                iid = f'{border["top"]}-{border["left"]}-{border["bottom"]}-{border["right"]}'
                prob = self.dd_wrapper.evaluate_model(im1, iid)
                if self.tag in prob:
                    prob = prob[self.tag]
                else:
                    prob = 0

                if (prob - best_prob) + tolerance > 0:

                    best_status = copy.deepcopy(border)
                    best_prob = prob
                    changed = True

                print('Debug: [{}, {}, {}, {}], {} vs {}'.format(
                      border["top"],
                      border["left"],
                      border["bottom"],
                      border["right"],
                      prob, best_prob
                ))

            current_borders = copy.deepcopy(best_status)

            if not changed:
                # Best was the previos model
                break


        values = {
            "top": int( (best_status["top"]/512) * self.pil_image.size[0]), 
            "left": int( (best_status["left"]/512) * self.pil_image.size[1]),
            "bottom": int( (best_status["bottom"]/512) * self.pil_image.size[0]),
            "right": int( (best_status["right"]/512) * self.pil_image.size[1]),
            "prob": best_prob,
        }

        self.drawer.title = f"Best_{self.tag}"
        self.drawer.crop(
            best_status["top"],
            best_status["left"],
            best_status["bottom"],
            best_status["right"],
            export=True,
            image_to_use="ORIG"
        )

        print(f"Evaluating: {values}")
        return [values]


class DeepDanbooruObjectRecognitionUtil:

    def __init__(
        self, 
        pil_image,
        minimal_threshold = 0.5,
        max_display = 10
    ):

        self.dd_wrapper = DeepDanbooruWrapper()

        self.minimal_threshold = minimal_threshold
        self.pil_image = pil_image
        self.max_display = int(max_display)
        self.request_uuid = str(uuid.uuid1())

        self.export_directory = os.path.join(shared.opts.outdir_extras_samples, "ddor")
        if not os.path.exists(self.export_directory):
            os.makedirs(self.export_directory)

        self.export_directory = os.path.join(self.export_directory, self.request_uuid)
        if not os.path.exists(self.export_directory):
            os.makedirs(self.export_directory)

    def create_heatmaps_util(self, tags, kernel_x, kernel_y, step_x, step_y, minimal_percentage):

        self.drawer = DeepDanbooruObjectDrawer(
            self.pil_image.copy(),
            f"Result-{time.time()}",
            self.export_directory
        )

        for tag in tags.split(","):

            if not tag.strip():
                continue

            dd_node = DeepDanbooruObjectRecognitionNode(
                self.dd_wrapper,
                self.pil_image,
                tag.strip().replace(" ", "_"),
                export_directory = self.export_directory
            )

            figures = dd_node.create_heatmaps(
                kernel_x,
                kernel_y,
                step_x,
                step_y,
                minimal_percentage
            )

            if not figures:
                continue

            for figure in figures:
                self.drawer.draw_rect(figure, f"{tag.strip().replace('_', ' ')}:\n{figure['prob']:.3f}")

        self.drawer.crop(0,0,512,512, export=True, image_to_use="RECT")
        return self.drawer.rect_pil_image

    def create_rects(self, tags, steps, subdivisions, tolerance):

        self.drawer = DeepDanbooruObjectDrawer(
            self.pil_image.copy(),
            f"Result-{time.time()}",
            self.export_directory
        )

        for tag in tags.split(","):

            if not tag.strip():
                continue

            dd_node = DeepDanbooruObjectRecognitionNode(
                self.dd_wrapper,
                self.pil_image,
                tag.strip().replace(" ", "_"),
                export_directory = self.export_directory
            )

            figures = dd_node.rect_tag(steps, subdivisions, tolerance)

            if not figures:
                continue

            for figure in figures:
                self.drawer.draw_rect(figure, f"{tag.strip().replace('_', ' ')}:{figure['prob']}")

        self.drawer.crop(0,0,512,512, export=True, image_to_use="RECT")
        return self.drawer.rect_pil_image

    # Core Methods
    def extract_tags(self):

        tag_probs = {}

        print("Extracting all tags")
        model_tags = self.dd_wrapper.evaluate_model(self.pil_image, "extract", self.minimal_threshold)
        model_tags = dict(sorted(model_tags.items(), key=lambda x: -x[1])[0:self.max_display])
        print(json.dumps(str(model_tags), indent=4))
        model_tags = list(model_tags.keys())

        return model_tags


def element_id_prefix(element_id):
    return f'deepdanbooru_object_recognition_{element_id}'


class DeepDanbooruObjectRecognitionScript():

    def __init__(self):
        
        self.source_image = None
        self.evaluate_btn = None
        self.tags = None
        self.result_image = None
        self.log_label = None
        self.override_chk = None

    def on_ui_tabs(self):

        with gr.Blocks(analytics_enabled=False) as ui_component:

            with gr.Row():

                with gr.Column(scale=1, elem_classes="source-image-col"):
                    self.source_image = gr.Image(type="pil", label="Source Image", interactive=True, elem_id=element_id_prefix("source_image"))

                with gr.Column(scale=1, elem_classes="other elements"):

                    with gr.Row():
                        self.threshold_ui = gr.Number(value=0.5, label="Threshold", elem_id=element_id_prefix("threshold_ui"), minimum=0, maximum=1)
                        self.max_display = gr.Number(value=10, label="Max to display on interrogate", elem_id=element_id_prefix("max_display_ui"), minimum=1, maximum=100)

                    self.interrogate_btn = gr.Button(value="Interrogate", elem_id=element_id_prefix("interrogate_btn"))
                    self.tags = gr.Textbox(value="1girl", label="Found tags", elem_id=element_id_prefix("tags_txt"))

                    with gr.Row():
                        self.steps = gr.Number(value=10, label="steps", elem_id=element_id_prefix("steps_ui"), minimum=5, maximum=100)
                        self.subdivisions = gr.Number(value=3, label="subdivisions", elem_id=element_id_prefix("subdivisions_ui"), minimum=3, maximum=50)
                        self.tolerance = gr.Number(value=0.05, label="tolerance", elem_id=element_id_prefix("tolerance_ui"), minimum=0, maximum=1)

                    self.evaluate_btn = gr.Button(value="Evaluate Method 1", elem_id=element_id_prefix("evaluete_btn"))

                    with gr.Row():
                        self.kernel_x = gr.Number(value=64, label="Kernel X", elem_id=element_id_prefix("kernel_x"), minimum=8, maximum=512)
                        self.kernel_y = gr.Number(value=64, label="Kernel Y", elem_id=element_id_prefix("kernel_y"), minimum=8, maximum=512)

                    with gr.Row():
                        self.step_x = gr.Number(value=32, label="Step X", elem_id=element_id_prefix("step_x"), minimum=8, maximum=512)
                        self.step_y = gr.Number(value=32, label="Step Y", elem_id=element_id_prefix("step_y"), minimum=8, maximum=512)
                    
                    with gr.Row():
                        self.minimal_percentage = gr.Number(value=0.85, label="minimal_percentage", elem_id=element_id_prefix("minimal_percentage_ui"), minimum=0, maximum=1)

                    self.evaluate_m2_btn = gr.Button(value="Evaluate Method 2", elem_id=element_id_prefix("evaluete_m2_btn"))

                with gr.Column(scale=1, elem_classes="result-image-col"):
                    self.result_image = gr.Image(type="pil", label="Result Image", interactive=False, elem_id=element_id_prefix("result_image"))

            with gr.Row():
                self.log_label = gr.Label(value="", label="Error", elem_id=element_id_prefix("log_label"))

            self.evaluate_btn.click(self.ui_click, inputs=[self.source_image, self.tags, self.threshold_ui, self.steps, self.subdivisions, self.tolerance], outputs=[self.result_image, self.log_label])
            self.evaluate_m2_btn.click(self.ui_click_m2, inputs=[self.source_image, self.tags, self.kernel_x, self.kernel_y, self.step_x, self.step_y, self.minimal_percentage], outputs=[self.result_image, self.log_label])
            self.interrogate_btn.click(self.ui_interrogate, inputs=[self.source_image, self.threshold_ui, self.max_display], outputs=[self.tags, self.log_label])
            return [(ui_component, "DeepDanboru Object Recognition", "deepdanboru_object_recg_tab")]


    def ui_interrogate(self, source_image_PIL, threshold_ui, max_display):

        # Init result image
        if not source_image_PIL:
            return None, "No source image found"

        dd_util = DeepDanbooruObjectRecognitionUtil(
            source_image_PIL,
            minimal_threshold=threshold_ui,
            max_display=max_display
        )

        dd_util.dd_wrapper.start()
        tag_probs = dd_util.extract_tags()
        dd_util.dd_wrapper.stop()

        return ", ".join(tag_probs), f"Complete request"


    def ui_click(self, source_image_PIL, tags, threshold_ui, steps, subdivisions, tolerance):

        # Init result image
        if not source_image_PIL:
            return None, "No source image found"

        dd_util = DeepDanbooruObjectRecognitionUtil(
            source_image_PIL
        )

        dd_util.dd_wrapper.start()
        result_image_PIL = dd_util.create_rects(
            tags,
            int(steps),
            int(subdivisions),
            tolerance
        )
        dd_util.dd_wrapper.stop()

        return result_image_PIL, f"Complete request: extra-images/ddor/{dd_util.request_uuid}"

    def ui_click_m2(self, source_image_PIL, tags, kernel_x, kernel_y, step_x, step_y, minimal_percentage):

        # Init result image
        if not source_image_PIL:
            return None, "No source image found"

        dd_util = DeepDanbooruObjectRecognitionUtil(
            source_image_PIL
        )

        dd_util.dd_wrapper.start()
        result_image_PIL = dd_util.create_heatmaps_util(
            tags,
            int(kernel_x),
            int(kernel_y),
            int(step_x),
            int(step_y),
            minimal_percentage
        )
        dd_util.dd_wrapper.stop()

        return result_image_PIL, f"Complete request: extra-images/ddor/{dd_util.request_uuid}"


ddors = DeepDanbooruObjectRecognitionScript()
script_callbacks.on_ui_tabs(ddors.on_ui_tabs)

# end of file
