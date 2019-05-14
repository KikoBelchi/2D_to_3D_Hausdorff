# CNN to obtain a 3D reconstruction of a towel from a single RGB image

### Dependencies

Create and activate a virtual environment in your project directory, and upgrade pip:
```
virtualenv -p /usr/bin/python3.5 2d_to3d_venv
source 2d_to3d_venv/bin/activate
pip install --upgrade pip
```
In my case, this installed Python3.5.2.

Install PyTorch and Tensorboard:

To install PyTorch, use the commands provided in https://pytorch.org/, which depend on your system.
E.g., for no Cuda, python 3.5, pip install on Linux:
```
pip3 install https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp35-cp35m-linux_x86_64.whl
pip3 install torchvision
```

The commands for installing Tensorboard also depend on your system.
For example, on Linux for CPU-only (no GPU), you would type these commands:
```
# pip install -U pip
pip install tensorflow
```
or
```
pip install --upgrade tensorflow
```

If you want to use this virtual environment for training the model, run:
```
pip install -r requirements_python352_py_for_training_in_visen2.txt
```

If instead you want to use this virtual environment for running the visualizations, run:
```
pip install -r requirements_python352_py_visualizations_in_leno2.txt

```

If instead you want to use this virtual environment for running the jupyter notebooks, run the following and create an environment in jupyter from this virtual environment:
```
requirements_python352_ipynb_no_pkg_resources.txt
```

### CAVEAT: Do not run these commands:
To let torch.device() be compatible with torch.load(..., map_location=torch.device()), 
you need to upgrade torch to version 0.4.1, but running the following will make the environment incompatible with the training code:
```
pip install --upgrade torch torchvision
```

### Train/Test Example
Train with default settings: 
```
python train.py 
```
Train, specifying some parameters, e.g.:
```
python train.py --num-epochs 30 --reordered-dataset 1 --submesh-num-vertices-vertical 9 --submesh-num-vertices-horizontal 9 --crop-centre-or-ROI 1 --dataset-number 3
```
Train, specifying some parameters in different lines:
```
python train.py \
--num-epochs 2 \
--log-interval 50
```


### Contact
For any doubt or upgrade contact Francisco Belchi: `frbegu at gmail.com`
