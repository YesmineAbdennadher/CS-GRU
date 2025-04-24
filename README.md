### Conda Environment Setting
```
conda create -n cs_gru 
conda activate cs_gru
conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch
```
### Spikingjelly Installation 
```
pip install spikingjelly

```
### Dataset
# NTIDIGITS
The data is hosted at https://www.dropbox.com/s/vfwwrhlyzkax4a2/n-tidigits.hdf5?dl=0.  
# SHD
The dataset is hosted at https://zenkelab.org/datasets/
# DVS Gesture
The DVS128 Gesture dataset does not support automatic download, but its resource_url_md5() function will print out the URL for obtaining the download address. The DVS128 Gesture dataset can be downloaded from https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794 to ./download.
# CIFAR10DVS
The dataset is hosted at https://figshare.com/articles/dataset/CIFAR10-DVS_New/4724671
### Training

*  Run the following command for the NTIDIGITS dataset
```
python -u train.py --epochs 300 --batch_size 128 --seed 1234 --T 10 --dataset 'NTIDIGITS'
