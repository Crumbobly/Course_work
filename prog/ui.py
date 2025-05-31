from typing import List

import numpy as np
import torch
from ultralytics import YOLO

from ai.unet.UNetModel import UNetModel
from processing.create_raw_data.create_image import create_image_from_file
from prog.analyze.create_defectograms import create_defectograms_1000, create_defectograms_500
from prog.analyze.unet_analyze import unet_predict
from prog.analyze.yolo_analyze import yolo_predict
import gradio as gr

from prog.formatting import get_result_for_all
from prog.resources.css import css

YOLO_MODEL = YOLO("./analyze/models/yolo/best.pt")
UNET_MODEL = UNetModel(in_channels=3, out_channels=4)
UNET_MODEL.load_state_dict(torch.load("./analyze/models/unet/unet_v2_epoch99.pth"))
UNET_MODEL.eval()


def analyze_yolo_file(file_obj):
    filename = file_obj.name

    data = np.loadtxt(filename).T
    defectograms_500 = create_defectograms_500(data)
    defectograms_1000 = create_defectograms_1000(data)
    defectogram_bbox = yolo_predict(defectograms_500, defectograms_1000, YOLO_MODEL)

    return (
        get_result_for_all(defectograms_1000, defectogram_bbox, "yolo"),
        defectograms_1000[0],
        defectograms_500[0],
        0,
        defectograms_1000,
        defectograms_500
    )


def analyze_unet_file(file_obj):
    filename = file_obj.name

    data = np.loadtxt(filename).T
    defectograms_500 = create_defectograms_500(data)
    defectograms_1000 = create_defectograms_1000(data)
    masks_raw, masks_post, bbox = unet_predict(defectograms_500, UNET_MODEL)
    print(masks_raw)
    print(masks_post)
    print(bbox)

    return (
        get_result_for_all(defectograms_1000, bbox, "unet"),
        defectograms_1000[0],
        defectograms_500[0],
        masks_raw[0],
        masks_post[0],
        0,
        defectograms_1000,
        defectograms_500,
        masks_raw,
        masks_post
    )


def change_page(index, direction, defectograms):
    new_index = max(0, min(index + direction, len(defectograms) - 1))
    new_defectogram = defectograms[new_index] if defectograms else None
    return new_defectogram, new_index


def change_page_500(index, direction, defectograms, masks_raw, masks_post):
    new_index = max(0, min(index + direction, len(defectograms) - 1))
    new_mask_raw = masks_raw[new_index] if masks_raw else None
    new_mask_post = masks_post[new_index] if masks_post else None
    return defectograms[new_index], new_mask_raw, new_mask_post, new_index


def model_selector_toggle_visibility_images(model_name):
    return gr.update(visible=(model_name == "UNet"))


def model_selector_toggle_visibility_button(model_name):
    if model_name == "YOLO":
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)


def clear(defectograms: List, defectograms_500: List, masks_raw: List, mask_post: List):
    defectograms.clear()
    defectograms_500.clear()
    masks_raw.clear()
    mask_post.clear()
    return 0, 0, "", None, None, None, None


def main():
    with gr.Blocks(css=css) as interface:
        gr.Markdown("# Анализ дефектограмм")

        # states
        state_defectogram_1000_list = gr.State([])
        state_defectogram_500_list = gr.State([])
        state_mask_raw_list = gr.State([])
        state_mask_post_list = gr.State([])
        page_index_1000 = gr.State(0)
        page_index_500 = gr.State(0)

        with gr.Row(equal_height=True):
            file_input = gr.File(label="Выберите файл", file_types=[".txt"])
            model_selector = gr.Dropdown(
                choices=["YOLO", "UNet"],
                value="YOLO",
                label="Модель"
            )

        analyze_button_YOLO = gr.Button("Анализировать Yolo")
        analyze_button_UNet = gr.Button("Анализировать UNet", visible=False)
        gr.Markdown("<br>")

        current_defectoram_500_image = gr.Image(label="Дефектограмма 500", type="pil", height="25px")
        current_defectoram_mask_raw_image = gr.Image(label="Маска RAW", type="pil", height="25px", visible=False)
        current_defectoram_mask_post_image = gr.Image(label="Маска", type="pil", height="25px", visible=False)

        with gr.Row():
            prev_btn_500 = gr.Button("←")
            defectogram_index_500 = gr.Text("0", interactive=False, show_label=False, container=False, elem_classes="text-center")
            next_btn_500 = gr.Button("→")

        gr.Markdown("<br>")
        file_result_text = gr.Text(label="Результат")
        current_defectoram_1000_image = gr.Image(label="Дефектограмма 1000", type="pil", height="25px")

        with gr.Row():
            prev_btn = gr.Button("←")
            defectogram_index_1000 = gr.Text("0", interactive=False, show_label=False, container=False, elem_classes="text-center")
            next_btn = gr.Button("→")

        page_index_500.change(
            fn=lambda x: str(x),
            inputs=[page_index_500],
            outputs=[defectogram_index_500]
        )
        page_index_1000.change(
            fn=lambda x: str(x),
            inputs=[page_index_1000],
            outputs=[defectogram_index_1000]
        )
        model_selector.change(
            fn=model_selector_toggle_visibility_images,
            inputs=model_selector,
            outputs=current_defectoram_mask_raw_image
        )
        model_selector.change(
            fn=model_selector_toggle_visibility_images,
            inputs=model_selector,
            outputs=current_defectoram_mask_post_image
        )
        model_selector.change(
            fn=model_selector_toggle_visibility_button,
            inputs=model_selector,
            outputs=[analyze_button_YOLO, analyze_button_UNet]
        )
        model_selector.change(
            fn=clear,
            inputs=[state_defectogram_1000_list, state_defectogram_500_list, state_mask_raw_list, state_mask_post_list],
            outputs=[page_index_500, page_index_1000,
                     file_result_text,
                     current_defectoram_500_image,
                     current_defectoram_1000_image, current_defectoram_mask_raw_image, current_defectoram_mask_post_image]
        )
        file_input.change(
            fn=clear,
            inputs=[state_defectogram_1000_list, state_defectogram_500_list, state_mask_raw_list, state_mask_post_list],
            outputs=[page_index_500, page_index_1000,
                     file_result_text,
                     current_defectoram_500_image,
                     current_defectoram_1000_image, current_defectoram_mask_raw_image, current_defectoram_mask_post_image]
        )

        analyze_button_YOLO.click(
            fn=analyze_yolo_file,
            inputs=[file_input],
            outputs=[
                file_result_text,
                current_defectoram_1000_image, current_defectoram_500_image,
                page_index_1000,
                state_defectogram_1000_list, state_defectogram_500_list
            ]
        )
        analyze_button_UNet.click(
            fn=analyze_unet_file,
            inputs=[file_input],
            outputs=[
                file_result_text,
                current_defectoram_1000_image, current_defectoram_500_image, current_defectoram_mask_raw_image, current_defectoram_mask_post_image,
                page_index_1000,
                state_defectogram_1000_list, state_defectogram_500_list, state_mask_raw_list, state_mask_post_list
            ]
        )

        prev_btn.click(fn=change_page,
                       inputs=[page_index_1000, gr.State(-1), state_defectogram_1000_list],
                       outputs=[current_defectoram_1000_image, page_index_1000])

        next_btn.click(fn=change_page,
                       inputs=[page_index_1000, gr.State(1), state_defectogram_1000_list],
                       outputs=[current_defectoram_1000_image,  page_index_1000])

        prev_btn_500.click(fn=change_page_500,
                           inputs=[page_index_500, gr.State(-1), state_defectogram_500_list, state_mask_raw_list, state_mask_post_list],
                           outputs=[current_defectoram_500_image, current_defectoram_mask_raw_image, current_defectoram_mask_post_image, page_index_500])

        next_btn_500.click(fn=change_page_500,
                           inputs=[page_index_500, gr.State(1), state_defectogram_500_list, state_mask_raw_list, state_mask_post_list],
                           outputs=[current_defectoram_500_image, current_defectoram_mask_raw_image, current_defectoram_mask_post_image, page_index_500])

    interface.launch()


main()
