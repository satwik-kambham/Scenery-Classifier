import json
from queue import Queue
from threading import Thread
from flask import Flask, Response, jsonify, render_template, request
from data_loader import load_images, TRAINING_DATA_PATH
from model_trainer import Classifier

app = Flask(__name__)
model = None
costs = Queue()
im = None

app.config['UPLOAD_FOLDER'] = '../images'

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

    batch_size = int(training_parameters['batch-size'])

    if batch_size > 0:
        images, labels = images[:batch_size], labels[:batch_size]

    global model
    model = Classifier.new_model(images, labels, training_parameters)

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

@app.route('/upload', methods = ['GET', 'POST'])
def upload():
    global im
    if request.method == 'POST':
      f = request.files['file']
      im = app.config['UPLOAD_FOLDER'] + '/' + f.filename
      f.save(im)
      return 'file uploaded successfully'

    return request.form.to_dict()

@app.route('/predict', methods = ['GET'])
def predict():
    global im
    label, prediction = model.test(im)
    c1, c2 = model.training_parameters['category-1'], model.training_parameters['category-2']
    prediction = prediction[0]
    # os.remove(im)
    # im = None
    data = {
        'label': c1 if label == 0 else c2,
        'prediction': prediction if label == 1 else 1 - prediction
    }
    print(data)
    return jsonify(data)

@app.route('/store', methods = ['GET', 'POST'])
def store():
    fname = json.loads(request.data.decode('utf-8'))['fname']
    print(fname)
    model.to_hdf('../models/'+fname)
    return 'Model stored'

@app.route('/load', methods = ['GET', 'POST'])
def load():
    global model
    fname = json.loads(request.data.decode('utf-8'))['fname']
    model = Classifier.from_hdf('../models/'+fname)
    return jsonify(model.training_parameters)
