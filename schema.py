from pydantic import BaseModel

class CarFeatures(BaseModel):
    engineSize: float
    year: int
    mileage: int
    fuelType: str

class PricePrediction(BaseModel):
    predicted_price: float
