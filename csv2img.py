#encoding=utf-8

from PIL import Image
import numpy as np

def csv2img(train_test):
    fp = open("/home/pan/Datasets/mnist/{}_data.csv".format(train_test))
    fp.readline()
    fout = open(train_test+"_labels.txt","w")
    for i, line in enumerate(fp):
        print(i)
        temp = line.strip().split(",")
        img = np.array(temp[1:]).astype(np.uint8)
        img = np.reshape(img,[28,28])
        img_name = train_test+"_datas/"+str(i).zfill(5)+".png"
        Image.fromarray(img).save(img_name)
        fout.write("{},{}\n".format(img_name, temp[0]))
    fp.close()
    fout.close()

if __name__ == "__main__":
    csv2img("train")
    csv2img("test")
