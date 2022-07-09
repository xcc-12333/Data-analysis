import joblib


class CarPriceModel:
    def __init__(self):
        self.scaler = None
        self.predictor = None

    def load_models(self):
        print("load_models")
        model_dir = "./models"
        self.scaler = joblib.load(f"{model_dir}/standardScaler.joblib")
        self.predictor = joblib.load(f"{model_dir}/random_model.joblib")

carPriceModel = CarPriceModel()
carPriceModel.load_models()