from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
import joblib

app = FastAPI()

model = joblib.load("mymodel_1")

templates = Jinja2Templates(directory="templates")


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "result": None,
            "url": None
        }
    )


@app.post("/predict")
async def predict_iris(
    request: Request,
    sl: float = Form(...),
    sw: float = Form(...),
    pl: float = Form(...),
    pw: float = Form(...)
):

    result = model.predict([[sl, sw, pl, pw]])[0]

    url = "https://daylily-phlox.eu/wp-content/uploads/2021/10/Iris-setosa-dwarf-form.jpg"

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "result": int(result),
            "url": url
        }
    )
