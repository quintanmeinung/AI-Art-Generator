import os, base64, requests
from .base import BaseImageProvider, ImageGenResult

API_URL = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"

class StabilityProvider(BaseImageProvider):
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("STABILITY_API_KEY")
        if not self.api_key:
            raise RuntimeError("STABILITY_API_KEY not set")

    def generate(self, prompt: str, *, size: str = "1024x1024",
                 negative_prompt=None, seed=None) -> ImageGenResult:
        w, h = map(int, size.split("x"))
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        text_prompts = [{"text": prompt, "weight": 1}]
        if negative_prompt:
            text_prompts.append({"text": negative_prompt, "weight": -1})

        payload = {
            "text_prompts": text_prompts,
            "cfg_scale": 7,
            "height": h,
            "width":  w,
            "samples": 1,
            "steps": 30
        }
        if seed is not None:
            payload["seed"] = int(seed)

        r = requests.post(API_URL, headers=headers, json=payload, timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f"Stability API {r.status_code}: {r.text}")

        b64 = r.json()["artifacts"][0]["base64"]
        return ImageGenResult(
            image_bytes=base64.b64decode(b64),
            provider_meta={"provider": "stability", "model": "sdxl-1.0"}
        )
