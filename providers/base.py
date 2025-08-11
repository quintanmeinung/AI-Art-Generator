from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class ImageGenResult:
    url: Optional[str] = None
    image_bytes: Optional[bytes] = None
    provider_meta: Dict[str, str] | None = None

class BaseImageProvider:
    def generate(
        self, prompt: str, *, size: str = "1024x1024",
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None
    ) -> ImageGenResult:
        raise NotImplementedError