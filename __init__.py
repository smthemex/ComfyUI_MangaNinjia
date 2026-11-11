"""ComfyUI_MangaNinjia - native ComfyUI port.

Legacy wrapper implementation (diffusers/transformers based) is archived under
_legacy/ and is intentionally NOT imported. The active implementation lives in
manganinjia_native/ and only depends on torch + ComfyUI itself.
"""

from .manganinjia_native.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
