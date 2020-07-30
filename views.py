from flask import Flask, request, jsonify
from flask import render_template
from datetime import datetime
from inference import test_sample,creat_model
from math import sqrt
from time import sleep
from . import app
import json
import cv2

model = creat_model()

@app.route("/",methods=['GET', 'POST'])

def home():
    if request.method == "POST":
        path = request.json["name"]
        print(path)
        #fileid = path.split(".")[0].split("_")[1]
        msk = test_sample(model)
        print(msk.shape)
        #regs = get_json(fileid)
        #with open('./static/{}.json'.format(fileid), 'w') as fp:
        #   json.dump(regs, fp)
        cv2.imwrite("./output.png", msk[:,:,None].repeat(3,axis=2))
        return {"msk":msk.tolist()}
        
    return render_template("new.html")

