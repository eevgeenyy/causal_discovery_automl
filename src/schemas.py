from pydantic import BaseModel
import models
from datetime import timedelta


class DatasetBase(BaseModel):
    dataset_link: str
    rows_number: int
    columns_number: int
    baseline: float
    #has_categorical: bool
   # has_numerical: bool
    best_run: str

class RunBase(BaseModel):
    dataset_id = int
    total_time = timedelta
    best_result = float
    best_algorithm = str
    best_params = str

