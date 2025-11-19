"""
Pydantic schemas for request validation.

Defines the structure of the JSON body expected by the /predict endpoint:
- PredictJSON: wraps the input feature data in a "data" field.
"""

from pydantic import BaseModel
from typing import List, Union


class PredictJSON(BaseModel):
    """
    JSON payload schema for predictions.

    The `data` field can be:
        - A single sample: List[float]      -> one row of 30 features
        - Multiple samples: List[List[float]] -> many rows of 30 features
    """
    # Raw feature data sent by the client; shape is validated later in preprocessing.
    data: Union[List[float], List[List[float]]]
