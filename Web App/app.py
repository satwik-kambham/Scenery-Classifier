from queue import Queue
from threading import Thread
from flask import Flask, Response, render_template, request
from data_loader import load_images, TRAINING_DATA_PATH
from model_trainer import Classifier

app = Flask(__name__)
model = None
costs = Queue()

def add_cost(epoch, cost):
    costs.put((epoch, cost))

@app.route('/')
def index():
    return render_template('index.html')


@app.route("/train", methods=['POST', 'GET'])
def train():
    training_parameters = request.form.to_dict()
    
    # Load training data
    images, labels = load_images(TRAINING_DATA_PATH, training_parameters['category-1'], training_parameters['category-2'])
    labels = labels.reshape(labels.shape[0], 1)

    images, labels = images[:50], labels[:50]

    global model
    model = Classifier(images, labels, training_parameters)

    p = Thread(target=model.train, args=(add_cost,))
    p.start()

    return render_template('train.html', training_parameters=training_parameters)


@app.route("/data")
def stream():
    def eventStream():
        if costs.empty():
            yield 'data: "Depleted"\n\n'
        else:
            epoch, cost = costs.get()
            yield 'data: {"epoch": '+str(epoch)+', "cost": '+str(cost)+'}\n\n'
    return Response(eventStream(), mimetype="text/event-stream")
            
