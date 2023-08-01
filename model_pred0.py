import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import glob as glob
from IPython.display import clear_output
import pathlib


img_width, img_height = 128, 96
batch_size = 64




class_names = ['1-1Phi0', '1-1Phi180','1-2Phi0','1-2Phi180','2-1Phi0','2-1Phi180', 'circular', 'undefined']

model = tf.keras.models.load_model("model2.h5")


ypredicted=[]
yscore=[]
fname=[]
counterrr = 0
fldr = "0"

files = glob.glob("./"+fldr+"/*.png")
for file in files:
    counterrr += 1
    img = file.split("/"+fldr)[1];     img = img[1:]
    #print(img)
    fname.append(img)
    img = tf.keras.utils.load_img(
        file, target_size=(img_height, img_width)
    )

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array,verbose=False)
    score = tf.nn.softmax(predictions[0])
    ypredicted.append(class_names[np.argmax(score)])
    yscore.append(np.max(score)*100)
    print(counterrr, flush=True)
    clear_output(wait=True)






C=[]
x0 =[]
y0 =[]
orbit_name = []
for img in fname:
    name = img.split("_")
    C.append(float(name[2][:-4]))
    x0.append(float(name[0][2:]))
    y0.append(float(0))
    orbit_name.append("df"+str(float(name[0][2:]))+"0.0"+"_"+name[2][:-4])

def vy0(C,p,x,y,vx=0):
    """
    Initial Velocity for cte jacobi conservation
    in:
        C = cte jacobi
        v = position and velocities of the particle
        cartesian coordinates
        p = mass ratio parameter == μ2
    out:
        ydot = velocity in y

    """
    μ = p
    #distance vectors
    r1 = np.sqrt((x+μ)**2 + y**2)
    r2 = np.sqrt((x - (1-μ))**2 + y**2)
    vy00 = np.sqrt(x**2 + y**2 - vx**2 + 2*((1-μ)/r1 + μ/r2) - C)
    if x > 0:
        return vy00
    if x < 0:
        return vy00

df = pd.DataFrame()
df['orbit'] = orbit_name
df['C'] = C
df['x0'] = x0
df['y0'] = y0
df['label'] = ypredicted
df['score'] = yscore
df['vx0'] = np.zeros(len(df.x0))
vy = []
for i in range(len(df.x0)):
    vy.append(vy0(df.C[i],0.001,df.x0[i],df.y0[i]))
df['vy0'] = vy


df.to_csv("./split_results/"+fldr+".csv",index=False)
