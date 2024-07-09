## Dependency
The code is tested on `python 3.8, Pytorch 1.13`.

##### Setup environment

```bash
conda create -n FSMNet python=3.8
source activate FSMNet # or conda activate FSMNet
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install einops h5py matplotlib scikit_image tensorboardX yacs pandas opencv-python timm ml_collections
```
