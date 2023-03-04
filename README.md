# ML-Lab

### Recomended Path for installing dependencies

Use **conda env** + **pip**

```powershell
conda create -n ml-labs python=3.7
conda activate ml-labs
```

- ML

```powershell
# for scikit-learn
pip install -U scikit-learn
# matplotlib
pip install matplotlib
```
scikit-learn: 1.0.2; matplotlib: 3.5.3

- DL

```powershell
# pytorch
# on cpu
pip install torch torchvision
# on gpu
# download following files from the link, please make sure the cuda version is supported by your driver
# https://download.pytorch.org/whl/cu117/torchvision-0.14.1%2Bcu117-cp37-cp37m-win_amd64.whl
# https://download.pytorch.org/whl/cu117/torch-1.13.1%2Bcu117-cp37-cp37m-win_amd64.whl
# change to your directory containing the 2 whl files, then run
pip install ./torch-1.13.1+cu117-cp37-cp37m-win_amd64.whl
pip install ./torchvision-0.14.1+cu117-cp37-cp37m-win_amd64.whl
# using conda to install pytorch is too much pain!
# you can test the installment of pytorch through pytorch_installation_test.py
```

Other dependencies will be installed on the fly.

### Other Dependencies

Other installments(if used), please list below(with specific version):

```
# requiements.txt
```

