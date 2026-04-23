from .StepByStep_Sampler import StepByStepSampler
from .StepByStep_Viewer import StepStepPlayer, StepStepComparer

NODE_CLASS_MAPPINGS = {
    "StepByStepSampler": StepByStepSampler,
    "StepStepPlayer": StepStepPlayer,
    "StepStepComparer": StepStepComparer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StepByStepGridSampler": "Step-by-Step Sampler",
    "StepStepPlayer": "Step-by-Step Player",
    "StepStepComparer": "Step-by-Step Comparer"
}

WEB_DIRECTORY = "./js"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]