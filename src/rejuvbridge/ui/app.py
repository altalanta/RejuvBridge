import gradio as gr


def _predict_stub(image):
    return image, {"example_metric": 0.42}


def launch_ui(host: str = "0.0.0.0", port: int = 7860):
    with gr.Blocks(title="RejuvBridge UI") as demo:
        gr.Markdown("# RejuvBridge — Imaging↔Omics Demo")
        with gr.Row():
            inp = gr.Image(type="pil", label="Tile")
            out_img = gr.Image(type="pil", label="Prediction Overlay")
        out_json = gr.JSON(label="Cross-modal metrics")
        btn = gr.Button("Run")
        btn.click(fn=_predict_stub, inputs=inp, outputs=[out_img, out_json])
    demo.launch(server_name=host, server_port=port)

