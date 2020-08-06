# flask-pytroch-cloud
This is a flask web app for deep learning glomerular segmentation and pathomic feature extraction on digital pathology image of donor kidney frozen section.

![image info](./static/webpage.png)

# Functions
For one whole slide image of kidney frozen section, you can:
- Choose which histology primitives to segment
- Have an overview statistic report for this slide 
- Click on one object detected and look at its detailed pathomic features 

# Tools
Flask - web framework<br> 
fabric.js - canvas<br>
PyTorch - models<br>
pyradiomics - pathomic features<br>
(Nginx - cloud)<br>
(Gunicorn - cloud)
# Installation
#### create a new conda env: 
`conda create -n [name of enviroment] python=3.7`
#### install packages: 
`pip install -r requirements.txt`

# Run flask app on local computer
`cd flask-pytorch-cloud 
conda activate [name of enviroment] 
python app.py`<br>
Then open the url and upload the sample image `im0.png` to test the model.

# Deploy on a cloud server



   
