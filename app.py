from flask import Flask, request, jsonify
from flask import render_template
from datetime import datetime
from inference import *
from math import sqrt
from time import sleep
import json
import cv2

app = Flask(__name__)

@app.route("/",methods=['GET', 'POST'])

def home():
    if request.method == "POST":
        path = request.json["name"]
        print(path)
        fileid = path.split(".")[0]
        model = creat_model()
        msk = test_sample(model)
        print(msk.shape)
        # save mask
        cv2.imwrite("./log/{}".format(path), msk[:,:,None].repeat(3,axis=2))
        # get contours and features for this slide
        regs = get_json(path)
        # save json
        with open('./static/{}.json'.format(fileid), 'w') as fp:
           json.dump(regs, fp)
        
        return {"msk":"done"}
        
    return render_template("new.html")

if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    # Flask's development server will automatically serve static files in
    # the "static" directory. See:
    # http://flask.pocoo.org/docs/1.0/quickstart/#static-files. Once deployed,
    # App Engine itself will serve those files as configured in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)