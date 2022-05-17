import os
import torch
import numpy as np
import argparse

seed = 123
np.random.seed(seed)


parser = argparse.ArgumentParser()

parser.add_argument("--dir", type=str, default="/scratch/allsamples_shhs",
                    help="File path to the PSG and annotation files.")

args = parser.parse_args()

dire = '/scratch/allsamples_shhs/'
data_dir = os.path.join(dire, "shhs_outputs")    #on gnode27 = "numpy_subjects"

files = os.listdir(data_dir)
files = np.array([os.path.join(data_dir, i) for i in files])
files.sort()


######## pretext files##########

pretext_files = list(np.random.choice(files,264,replace=False))    #change

print("pretext files: ", len(pretext_files))
from tqdm import tqdm
# load files
half_window = 3

os.makedirs(dire+"/pretext/",exist_ok=True)

cnt = 0
for file in tqdm(pretext_files):
    x_dat = np.load(file)["x"]
    if x_dat.shape[-1]==2 or x_dat.shape[-1]==1:
        x_dat = x_dat.transpose(0,2,1)
    for i in range(half_window,x_dat.shape[0]-half_window):
        dct = {}
        temp_path = os.path.join(dire+"/pretext/",str(cnt)+".npz")
        dct['pos'] = x_dat[i-half_window:i+half_window+1]
        np.savez(temp_path,**dct)
        cnt+=1


######## test files##########
test_files = sorted(list(set(files)-set(pretext_files))) 
os.makedirs(dire+"/test/",exist_ok=True)

print("test files: ", len(test_files))

for file in tqdm(test_files):
    new_dat = dict()
    dat = np.load(file)

    if dat['x'].shape[-1]==2:
        new_dat['x'] = dat['x'].transpose(0,2,1)
    else:
        new_dat['x'] = dat['x']

    new_dat['y'] = dat['y']
    
    temp_path = os.path.join(dire+"/test/",os.path.basename(file))
    np.savez(temp_path,**new_dat)
