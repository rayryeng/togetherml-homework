from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from starlette.config import Config
import uvicorn
import os
from fastai import *
from fastai.vision import *
import urllib
import aiohttp # Added by Ray

app = Starlette(debug=True)

app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['*'], allow_methods=['*'])

### EDIT CODE BELOW ###

answer_question_1 = """ 
Overfitting is when your model performs extremely well on the training set and
very poorly on the validation datset.  It essentially memorized individual
training examples which therefore cannot generalize to the actual problem and
thus fails on unseen examples.   Underfitting is the exact opposite.  
The model cannot generalize for both training and validation datasets. 
In other words, both training loss and validation loss are similar, but do not
decrease over the epochs.
"""

answer_question_2 = """ 
The objective of training a neural network is to find the parameters that best
minimize a loss function.  In our case, the loss function for this task is
the cross-entropy loss function.  Because the loss function is highly non-linear,
we use gradient descent to help us determine what the minimum of the loss
function is and thus the parameters that correspond to this loss.  Starting
at some initial values, we calculate the gradient at these values for every
parameter in the network.  The gradient tells us in which direction we
would need to proceed in order to get to the minimum.  The magnitude of the
step we need to take is defined by the learning rate.  For each iteration,
we take a step towards the minimum which is guided by the learning rate.
Too small a learning rate means it will take us forever to get to the minimum
and too large a learning rate would mean we would possibly overshoot the
minimum and possibly diverge away from it.  We keep iterating and moving
towards the minimum until the gradient norm is close to 0, meaning that
we can no longer improve our solution and thus we have reached a minimum.
The resulting parameters that provide this minimum are the ones we use
to represent the neural network.
"""

answer_question_3 = """ 
The goal of regression is to predict continuous-valued output.  This particular
task is classification, which seeks to provide an output belonging to one of
a small number of labels.  Regression would be used for predicted housing
prices, predicting the height of your child or the temperature of a city.  The
only difference would be the final output layer.  Instead of a softmax layer,
we'd use a single linear neuron.  In addition, the ground truth values
in the dataset would need to be continuous-valued as well.
"""

## Replace none with your model
pred_model = load_learner('./models', file='tng_model.pkl')

# Added by Ray
async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()

@app.route("/api/answers_to_hw", methods=["GET"])
async def answers_to_hw(request):
    return JSONResponse([answer_question_1, answer_question_2, answer_question_3])

@app.route("/api/class_list", methods=["GET"])
async def class_list(request):
    return JSONResponse(['beverly-crusher', 'data', 'deanna-troi', 'geordi-la-forge',
                          'jean-luc-picard', 'katherine-pulaski', 'tasha-yar',
                          'wesley-crusher', 'will-riker', 'worf'])

@app.route("/api/classify", methods=["POST"])
async def classify_url(request):
    body = await request.json()
    url_to_predict = body["url"]

    ## Make your prediction and store it in the preds variable
    bytes = await get_bytes(url_to_predict)
    img = open_image(BytesIO(bytes))
    preds, _, _ = pred_model.predict(img)

    return JSONResponse({
        "predictions": preds,
    })

### EDIT CODE ABOVE ###

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=int(os.environ['PORT']))
