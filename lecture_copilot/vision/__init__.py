from lecture_copilot.vision.change import difference, signature
from lecture_copilot.vision.extract import (
    OpenRouterVisionExtractor,
    SlideContent,
    TesseractExtractor,
    build_extractor,
    image_to_data_uri,
    parse_slide_content,
)
from lecture_copilot.vision.screen import Monitor, ScreenCapture, list_monitors
from lecture_copilot.vision.watcher import SlideWatcher

__all__ = [
    "Monitor",
    "OpenRouterVisionExtractor",
    "ScreenCapture",
    "SlideContent",
    "SlideWatcher",
    "TesseractExtractor",
    "build_extractor",
    "difference",
    "image_to_data_uri",
    "list_monitors",
    "parse_slide_content",
    "signature",
]
