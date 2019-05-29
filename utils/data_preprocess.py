import urllib.request
import gzip
import pickle
import os
import numpy as np
from PIL import Image


dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/mnist.pkl"
url_base = 'http://yann.lecun.com/exdb/mnist/'
data_file = {
    'train_img':'train-images-idx3-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
}
lable_file = {
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz' 
}

def _download_data(file_name):
    print(f"File url: {url_base + file_name}")
    file_path = dataset_dir + "/" + file_name        
    urllib.request.urlretrieve(url_base + file_name, file_path)
    print(f"Download file: {file_path}")
    
def _convert_data(file_name, dtype='data', img_size=784):
    file_path = dataset_dir + "/" + file_name
    print(f"Converting file {file_path} to Numpy array")
    
    with gzip.open(file_path, 'rb') as f:
        if dtype == 'lable':
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        elif dtype == 'data':
            data = np.frombuffer(f.read(), np.uint8, offset=16)
            data = data.reshape(-1, img_size)
            
    return data

def _init_data():
    data_set = {}
    
    for name, file in data_file.items():
        _download_data(file)
        data_set[name] = _convert_data(file, dtype='data')
    for name, file in lable_file.items():
        _download_data(file)
        data_set[name] = _convert_data(file, dtype='lable')
        
    with open(save_file, 'wb') as f:
        pickle.dump(data_set, f, -1)
    print(f"Save data set: {save_file}")
    
def get_data():
    if not os.path.exists(save_file):
        _init_data()

    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)
        
    return dataset

def one_hot_label(lable_int):
    mat = np.zeros((lable_int.size, 10))
    for idx, row in enumerate(mat):
        row[lable_int[idx]] = 1

    return mat

def reshape_data(data):
    data = data.reshape(-1, 28, 28)

    return data

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()