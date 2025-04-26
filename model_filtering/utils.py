import torch
import numpy as np
from datetime import datetime
from rich.console import Console
from rich.theme import Theme

# --------------------------------------------------------------------------- #
# Helper: make *anything* JSON-serialisable                                     #
# --------------------------------------------------------------------------- #
def json_default(obj):
    """Fallback encoder for json.dump."""
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.dim() == 0 else obj.tolist()
    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.item() if np.ndim(obj) == 0 else obj.tolist()
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"{type(obj).__name__} is not JSON serialisable")

# --------------------------------------------------------------------------- #
# Rich console setup                                                            #
# --------------------------------------------------------------------------- #
custom_theme = Theme(
    {
        "info": "cyan",
        "success": "green",
        "warning": "yellow",
        "error": "bold red",
        "highlight": "bold magenta",
        "metric": "bold cyan",
        "time": "bold blue",
    }
)
console = Console(theme=custom_theme)