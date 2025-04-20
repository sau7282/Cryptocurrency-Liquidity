import numpy as np

def make_predictions(features, model):
    # Assuming features are in a pandas DataFrame format
    prediction = model.predict(features)
    print(prediction)
    return prediction
