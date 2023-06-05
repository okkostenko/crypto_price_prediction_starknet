from typing import Union
from pydantic import BaseModel
from firebase_admin.firestore import Timestamp

class StatusDocument(BaseModel):
    current_usdt: str
    current_weth: str
    net_worth: str
    timestamp: Timestamp

class TradesDocument(BaseModel):
    action_type: Union["BUY", "SELL", "HOLD"]
    amount: str
    timestamp: Timestamp