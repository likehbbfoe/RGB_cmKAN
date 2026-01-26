from pydantic import BaseModel
from enum import Enum


class DataType(str, Enum):
    volga2k = 'volga2k'
    volga2k_unpaired = 'volga2k_unpaired'
    five_k = 'five_k' 
    samsung2iphone = 'samsung2iphone'
    samsung2iphone_unpaired = 'samsung2iphone_unpaired' 


class DataPathes(BaseModel):
    source: str
    target: str


class Data(BaseModel):
    type: DataType = DataType.volga2k
    train: DataPathes
    val: DataPathes
    test: DataPathes