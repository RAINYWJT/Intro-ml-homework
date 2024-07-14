import os
import pickle
import numpy as np
from learnware.model import BaseModel

class MyModel(BaseModel):
    def __init__(self):
        super(MyModel, self).__init__(input_shape=(561,), output_shape=(1,))
        dir_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(dir_path, "stacking_classifier_model.pkl")
        if not os.path.exists(model_path):
            print(f"File not found: {model_path}")
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def finetune(self, X: np.ndarray, y: np.ndarray):
        pass
    