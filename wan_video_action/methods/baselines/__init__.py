from .dino import SPEC as DINO
from .standard import SPEC as STANDARD
from .lora import SPEC as LORA
from .ttt import SPEC as TTT

SPECS = (STANDARD, LORA, TTT, DINO)

__all__ = ["DINO", "STANDARD", "LORA", "TTT", "SPECS"]
