"""
    @author: Jibaku789
    @version: 1.0
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
from PIL import ImageDraw, Image

from modules import script_callbacks
from modules import devices, images
from modules.deepbooru import DeepDanbooru
from modules import shared

class DeepDanbooruWrapper:

    def __init__(
        self,
        minimal_threshold = 0.5,
    ):

        self.dd_classifier = DeepDanbooru()
        self.minimal_threshold = minimal_threshold

    def start(self):
        print("Starting DeepDanboru")
        self.dd_classifier.start()

    def stop(self):
        print("Stopping DeepDanboru")
        self.dd_classifier.stop()

    def evaluate_model(self, pil_image):

        # Input image should be 512x512 before reach this point

        pic = images.resize_image(0, pil_image.convert("RGB"), 512, 512)

        #pic = pil_image.convert("RGB")
        a = np.expand_dims(np.array(pic, dtype=np.float32), 0) / 255

        with torch.no_grad(), devices.autocast():
            x = torch.from_numpy(a).to(devices.device)
            y = self.dd_classifier.model(x)[0].detach().cpu().numpy()

        probability_dict = {}
        for tag, probability in zip(self.dd_classifier.model.tags, y):

            if probability < self.minimal_threshold:
                continue

            if tag.startswith("rating:"):
                continue

            probability_dict[tag] = probability

        return probability_dict

class DeepDanbooruObjectDrawer:

    def __init__(
        self,
        pil_image,
        title,
        export_directory
    ):

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

    def crop(self, top, left, bottom, right, export=False):

        x1 = left
        x2 = right
        y1 = top
        y2 = bottom

        target_size = (512, 512)
        im1 = self.pil_image.resize(target_size)
        im2 = im1.crop((x1, y1, x2, y2))

        background = Image.new('RGB', target_size)
        background.paste(im2, (x1, y1))

        if export:
            prefix = f"{self.export_directory}/{self.title}_{top}_{left}_{bottom}_{right}"
            background.save(f"{prefix}.png")

        return background


    def draw_rect(self, borders, title):

        target_size = (512, 512)
        im1 = self.pil_image.resize(target_size)
        draw = ImageDraw.Draw(im1)
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
            0
        )

        draw.rectangle(
            [
                (borders["left"], borders["top"]), 
                (borders["right"], borders["bottom"])
            ],
            outline=color
        )

        draw.text(
            (borders["left"], borders["top"]),
            title,
            fill=color
        )

        self.pil_image = im1

class DeepDanbooruObjectRecognitionNode:

    def __init__(
        self,
        dd_wrapper,
        pil_image,
        tag,
        export_directory,
        steps = 10,
        subdivisions = 3,
        tolerance = 0.05
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

        self.steps = steps
        self.subdivisions = subdivisions
        self.tolerance = tolerance


    def rect_tag(self):

        # Init prob
        print(f"Evaluating: {self.tag}")

        im0 = self.drawer.crop(0, 0, 512, 512)
        initial_prob = self.dd_wrapper.evaluate_model(im0)

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

        for s in range(0, self.steps):

            # Calculate proportions
            changed = False

            x_diff = current_borders["right"] - current_borders["left"]
            y_diff = current_borders["bottom"] - current_borders["top"]

            propotions = {
                "top": int(current_borders["top"] + (y_diff * ((self.subdivisions-1)/self.subdivisions))),
                "left": int(current_borders["left"] + (x_diff * ((self.subdivisions-1)/self.subdivisions))),
                "bottom": int(current_borders["bottom"] - (y_diff * ((self.subdivisions-1)/self.subdivisions))),
                "right": int(current_borders["right"] - (x_diff * ((self.subdivisions-1)/self.subdivisions)))
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
                    border["right"],
                    export=False
                )

                prob = self.dd_wrapper.evaluate_model(im1)
                if self.tag in prob:
                    prob = prob[self.tag]
                else:
                    prob = 0

                if (prob - best_prob) + self.tolerance > 0:

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
            export=True
        )

        print(f"Evaluating: {values}")
        return values

    def rect_tag2(self):

        # Init prob
        print(f"Evaluating: {self.tag}")

        im1 = self.drawer.crop(0, 0, 512, 512)
        initial_prob = self.dd_wrapper.evaluate_model(im1)

        if self.tag not in initial_prob:
            return None

        initial_prob = initial_prob[self.tag]

        status = {
            "top": {
                "pos": 0,
                "last": initial_prob,
                "ban": False
            },
            "left": {
                "pos": 0,
                "last": initial_prob,
                "ban": False
            },
            "bottom": {
                "pos": 512,
                "last": initial_prob,
                "ban": False
            },
            "right": {
                "pos": 512,
                "last": initial_prob,
                "ban": False
            }
        }

        best_prob = initial_prob
        best_status = copy.deepcopy(status)

        im1 = self.drawer.crop(
            status["top"]["pos"],
            status["left"]["pos"],
            status["bottom"]["pos"],
            status["right"]["pos"]
        )

        count = -1

        while (status["top"]["pos"] < status["bottom"]["pos"]) and \
              (status["left"]["pos"] < status["right"]["pos"]): 

            count += 1

            if status["top"]["ban"] and \
               status["left"]["ban"] and \
               status["bottom"]["ban"] and \
               status["right"]["ban"]:
                # Best found
                break

            to_change_pos = list(status.keys())[int(count % 4)]
            to_change_mini_status = status[to_change_pos]

            if to_change_mini_status["ban"]:
                continue

            # Update to compare
            if to_change_pos in ["top", "left"]:
                to_change_mini_status["pos"] += self.step
            else:
                to_change_mini_status["pos"] -= self.step

            # Evaluate
            im1 = self.drawer.crop(
                status["top"]["pos"],
                status["left"]["pos"],
                status["bottom"]["pos"],
                status["right"]["pos"]
            )

            prob = self.dd_wrapper.evaluate_model(im1)
            if self.tag in prob:
                prob = prob[self.tag]
            else:
                prob = 0

            if (prob - to_change_mini_status["last"])+0.05 > 0:

                if prob > best_prob:
                    best_prob = prob
                    best_status = copy.deepcopy(status)

                to_change_mini_status["last"] = prob

            else:

                # Restore previos values and ban
                if to_change_pos in ["top", "left"]:
                    to_change_mini_status["pos"] -= self.step
                else:
                    to_change_mini_status["pos"] += self.step

                to_change_mini_status["ban"] = True

            print('Debug: [{}, {}, {}, {}], {} vs {}, [{}, {}, {}, {}]'.format(
                  status["top"]["pos"],
                  status["left"]["pos"],
                  status["bottom"]["pos"],
                  status["right"]["pos"],
                  prob, best_prob, 
                  status["top"]["ban"],
                  status["left"]["ban"],
                  status["bottom"]["ban"],
                  status["right"]["ban"]
            ))

        # Condensate
        values = {
            "top": int( (best_status["top"]["pos"]/512) * self.pil_image.size[0]), 
            "left": int( (best_status["left"]["pos"]/512) * self.pil_image.size[1]),
            "bottom": int( (best_status["bottom"]["pos"]/512) * self.pil_image.size[0]),
            "right": int( (best_status["right"]["pos"]/512) * self.pil_image.size[1]),
            "prob": best_prob,
        }

        self.drawer.title = f"Best_{self.tag}"
        self.drawer.crop(
            best_status["top"]["pos"],
            best_status["left"]["pos"],
            best_status["bottom"]["pos"],
            best_status["right"]["pos"]
        )

        print(f"Evaluating: {values}")
        return values


class DeepDanbooruObjectRecognitionUtil:

    def __init__(
        self, 
        pil_image,
        minimal_threshold = 0.5,
        max_display = 10,
        steps = 10,
        subdivisions = 3,
        tolerance = 0.05
    ):

        self.dd_wrapper = DeepDanbooruWrapper(
            minimal_threshold,
        )

        self.pil_image = pil_image
        self.max_display = int(max_display)
        self.steps = steps
        self.subdivisions = subdivisions
        self.tolerance = tolerance

        self.export_directory = os.path.join(shared.opts.outdir_extras_samples, "ddor")
        if not os.path.exists(self.export_directory):
            os.makedirs(self.export_directory)

    def create_rects(self, tags):

        figures = {}

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
                tag.strip(),
                export_directory = self.export_directory,
                steps = self.steps,
                subdivisions = self.subdivisions
            )

            figure = dd_node.rect_tag()
            figures[tag] = figure

            if not figure:
                continue

            self.drawer.draw_rect(figure, f"{tag.strip()}-{figure['prob']}")

        self.drawer.crop(0,0,512,512, export=True)
        return self.drawer.pil_image

    # Core Methods
    def extract_tags(self):

        tag_probs = {}

        print("Extracting all tags")
        model_tags = self.dd_wrapper.evaluate_model(self.pil_image)
        model_tags = dict(sorted(model_tags.items(), key=lambda x: -x[1])[0:self.max_display])
        print(json.dumps(str(model_tags), indent=4))
        model_tags = list(model_tags.keys())

        return model_tags

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
                    self.source_image = gr.Image(type="pil", label="Source Image", interactive=True, elem_id="source_image")

                with gr.Column(scale=1, elem_classes="other elements"):

                    with gr.Row():
                        self.threshold_ui = gr.Number(value=0.5, label="Threshold", elem_id="threshold_ui", minimum=0, maximum=1)
                        self.max_display = gr.Number(value=10, label="Max to display on interrogate", elem_id="max_display_ui", minimum=1, maximum=100)

                    self.interrogate_btn = gr.Button(value="Interrogate", elem_id="interrogate_btn")
                    self.tags = gr.Textbox(value="1girl", label="Found tags", elem_id="tags_txt")

                    with gr.Row():
                        self.steps = gr.Number(value=10, label="steps", elem_id="steps_ui", minimum=5, maximum=100)
                        self.subdivisions = gr.Number(value=3, label="subdivisions", elem_id="subdivisions_ui", minimum=3, maximum=50)
                        self.tolerance = gr.Number(value=0.05, label="tolerance", elem_id="tolerance_ui", minimum=0, maximum=1)

                    self.evaluate_btn = gr.Button(value="Evaluate", elem_id="evaluete_btn")

                with gr.Column(scale=1, elem_classes="result-image-col"):
                    self.result_image = gr.Image(type="pil", label="Result Image", interactive=False, elem_id="result_image")

            with gr.Row():
                self.log_label = gr.Label(value="", label="Error", elem_id="log_label")

            self.evaluate_btn.click(self.ui_click, inputs=[self.source_image, self.tags, self.threshold_ui, self.steps, self.subdivisions, self.tolerance], outputs=[self.result_image, self.log_label])
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

        return ", ".join(tag_probs), "Complete"


    def ui_click(self, source_image_PIL, tags, threshold_ui, steps, subdivisions, tolerance):

        # Init result image
        if not source_image_PIL:
            return None, "No source image found"

        dd_util = DeepDanbooruObjectRecognitionUtil(
            source_image_PIL,
            minimal_threshold=threshold_ui,
            steps=int(steps), 
            subdivisions=int(subdivisions),
            tolerance=tolerance
        )

        dd_util.dd_wrapper.start()
        result_image_PIL = dd_util.create_rects(tags)
        dd_util.dd_wrapper.stop()

        return result_image_PIL, "Complete"


ddors = DeepDanbooruObjectRecognitionScript()
script_callbacks.on_ui_tabs(ddors.on_ui_tabs)
