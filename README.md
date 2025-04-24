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
### Training

*  Run the following command for the NTIDIGITS dataset
```
python -u train.py --epochs 300 --batch_size 128 --seed 1234 --T 10 --dataset 'NTIDIGITS'
