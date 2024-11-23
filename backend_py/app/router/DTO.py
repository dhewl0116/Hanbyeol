
from pydantic import BaseModel


class Calculator_DTO(BaseModel):
    NumberOfDependents:int
    MaritalStatus:str
    MonthlyIncome:int
    AssetValue:int
    TotalDebt:int

class text_DTO(BaseModel):
    user_id:str
    sentence:str

class BNPL_DTO(BaseModel):
    user_id: str
    account_num: str
    money: int

class Macthing_DTO(BaseModel):
    sentence:str

class BNPL_return_DTO(BaseModel):
    user_id: str
    money: int

