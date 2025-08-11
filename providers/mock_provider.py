from .base import BaseImageProvider, ImageGenResult

PLACEHOLDER = "https://picsum.photos/seed/demo/1024/1024"

class MockProvider(BaseImageProvider):
    def generate(self, prompt: str, *, size: str = "1024x1024",
                 negative_prompt=None, seed=None) -> ImageGenResult:
        return ImageGenResult(
            url=PLACEHOLDER,
            provider_meta={"provider": "mock", "note": "placeholder image"}
        )