import json
import numpy as np
from typing import Tuple


def serialize_state(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> str:
    """
    Serializes (x, y, z) to a JSON string.
    """
    state = {"x": x.tolist(), "y": y.tolist(), "z": z.tolist()}
    return json.dumps(state)


def deserialize_state(json_str: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Deserializes JSON string back into (x, y, z).
    """
    data = json.loads(json_str)
    x = np.array(data["x"], dtype=np.float32)
    y = np.array(data["y"], dtype=np.float32)
    z = np.array(data["z"], dtype=np.float32)
    return x, y, z
