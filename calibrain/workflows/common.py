from pathlib import Path
from typing import Any, Dict

import runpy


def load_python_config(config_path: Path | str) -> Dict[str, Any]:
    namespace = runpy.run_path(str(config_path))
    if "CONFIG" not in namespace:
        raise ValueError(f"Config {config_path} must define a CONFIG dict.")
    return namespace["CONFIG"]
