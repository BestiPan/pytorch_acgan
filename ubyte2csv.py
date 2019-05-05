# coding=utf-8

import struct
from array import array
import numpy as np
import pandas as pd

def read_label():
    with open("t10k-labels.idx1-ubyte","rb") as fp:
        magic, size = struct.unpack(">II", fp.read(8))
        print(magic, size)
        label_data = array("B", fp.read())
        labels = []
        for i in list(range(size)):
            labels.append(label_data[i])
    labels = np.array(labels).reshape([-1,1])
    return labels

def read_image():
    with open("t10k-images.idx3-ubyte","rb") as fp:
        magic, size, rows, cols = struct.unpack(">IIII", fp.read(16))
        print(magic, size, rows, cols)
        image_data = array("B", fp.read())
        images = []
        for i in list(range(size)):
            images.append([0]*rows*cols)
        for i in list(range(size)):
            images[i][:] = image_data[i*rows*cols:(i+1)*rows*cols]
    images = np.array(images)
    return images

def read_file():
    images = read_image()
    labels = read_label()
    datasets = np.concatenate((labels,images),axis=1)
    datasets = pd.DataFrame(datasets,dtype=int)
    datasets.to_csv("test_data.csv",index=None)

if __name__ == "__main__":
    read_file()