import os, io, requests
from dotenv import load_dotenv
from PIL import Image, ImageDraw
import gradio as gr

from providers.mock_provider import MockProvider
try:
    from providers.stability_provider import StabilityProvider
except Exception:
    StabilityProvider = None

load_dotenv()  # loads STABILITY_API_KEY and PROVIDER from .env

prov_name = os.getenv("PROVIDER", "mock").lower()
if prov_name == "stability" and StabilityProvider is not None:
    provider = StabilityProvider()  # reads key from env
else:
    provider = MockProvider()

def generate_image(prompt: str, size: str = "1024x1024",
                   negative_prompt: str | None = None,
                   seed: int | None = None):
    # Step A: call provider and catch API errors
    try:
        result = provider.generate(
            prompt, size=size, negative_prompt=negative_prompt, seed=seed
        )
    except Exception as e:
        # e.g., "Stability API 401: ..." / "429: rate limit" / etc.
        return None, f"Provider error: {e}"

    # Step B: turn result into an image (bytes preferred, then URL, then fallback)
    try:
        if getattr(result, "image_bytes", None):
            img = Image.open(io.BytesIO(result.image_bytes)).convert("RGB")
            source = "stability (bytes)"
        else:
            r = requests.get(result.url, timeout=10)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            source = "url fetch"
        meta = (result.provider_meta or {})
        return img, f"provider: {meta.get('provider','?')} | model: {meta.get('model','?')} | source: {source}"
    except Exception as e:
        return None, f"Image assembly error: {e}"


with gr.Blocks(title="AI Art Generator") as demo:
    gr.Markdown("## AI Art Generator\nUsing provider from .env (`PROVIDER`).")
    with gr.Row():
        prompt = gr.Textbox(label="Prompt", scale=3, placeholder="A serene watercolor landscape at sunset")
        size = gr.Dropdown(choices=["512x512","768x768","1024x1024"], value="768x768", label="Size", scale=1)
    with gr.Row():
        negative = gr.Textbox(label="Negative prompt (optional)", placeholder="low quality, blurry", scale=3)
        seed = gr.Number(label="Seed (optional int)", precision=0, scale=1)
    btn = gr.Button("Generate", variant="primary")
    out = gr.Image(type="pil", label="Result")
    status = gr.Textbox(label="Status", interactive=False)
    btn.click(generate_image, [prompt, size, negative, seed], [out, status])

if __name__ == "__main__":
    demo.launch()

