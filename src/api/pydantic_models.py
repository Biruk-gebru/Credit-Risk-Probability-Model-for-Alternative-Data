from pydantic import BaseModel
from typing import Optional

class CreditRiskInput(BaseModel):
    ProviderId: str
    ProductId: str
    ProductCategory: str
    ChannelId: str
    Amount: float
    Value: float
    PricingStrategy: str
    TransactionHour: int
    TransactionDay: int
    TransactionMonth: int
    TransactionYear: int
    Amount_sum: float
    Amount_mean: float
    Amount_count: float
    Amount_std: float
    Value_sum: float
    Value_mean: float
    Value_count: float
    Value_std: float
    FraudResult: int

class CreditRiskOutput(BaseModel):
    prediction: int
    probability: float
