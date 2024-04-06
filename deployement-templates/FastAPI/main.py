import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi import APIRouter, FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI, File, HTTPException, UploadFile

from typing import Any
from pydantic import BaseModel, Field, ValidationError
import pickle
import pandas as pd
 
class Model (BaseModel):
    fixed_acidity: float  = Field(alias='fixed acidity')
    volatile_acidity: float  = Field(alias='volatile acidity')
    citric_acid : float  = Field(alias='citric acid')
    residual_sugar: float = Field(alias='residual sugar')
    chlorides: float
    free_sulfur_dioxide: int = Field(alias='free sulfur dioxide')
    total_sulfur_dioxide: int = Field(alias='total sulfur dioxide')
    density: float
    pH: float
    sulphates: float 
    alcohol: float

def json_verify(data_in) -> list[Any]:
    try:
        Model(**data_in)
        return(True)
    except ValidationError as e:
        return(e.json())
            
def load_model(path:str) -> object:
    return pickle.load(open(path, 'rb'))

MODEL_REV = 100
MODEL_NAME = 'wine-quality'
model = load_model(f'models/{MODEL_NAME}_v{MODEL_REV}.pkl')

app = FastAPI()

@app.get('/')
async def index():
    # Index Page - Landing Page for your model
    app.mount("/", StaticFiles(directory="landing", html = True), name="landing")
    return FileResponse('landing/index.html')

@app.get('/favicon.ico')
async def favicon():
    #Favicon for your model page
    return FileResponse('landing/src/images/favicon.png')

@app.post('/test', description= "Returns the message you send")
async def hello_world(text_msg:str):
    print('Your Message -> ',text_msg)
    return {'message':text_msg}

@app.post(f"/predict/{MODEL_NAME}", description=f'Retuns Prediction & Probability (for Classification) using Model - "{MODEL_NAME}_v{MODEL_REV}"')
async def inference(data_in: Model):

    data = list(data_in.dict().values())
    prediction = model.predict([data])
    try:
        probability = model.predict_proba([data])
    except:
        probability = 'NA'
    return {
        'prediction': prediction[0],
        'probability': probability
    }

@app.post(f'/predict-batch/{MODEL_NAME}', description=f'Retuns List of Prediction using Model - "{MODEL_NAME}_v{MODEL_REV}"')
async def batch_inference(csv_file: UploadFile = File(...)):
    print('Processing ', csv_file.filename)
    df = pd.read_csv(csv_file.file)                         #Read the batch file
    df.drop(['quality'],axis =1, inplace = True)   
    prediction = model.predict(df)   
    print(prediction)                   
    return {
        'prediction': prediction,
    }

if __name__=="__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)
