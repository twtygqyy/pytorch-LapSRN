# PyTorch LapSRN
Implementation of CVPR2017 Paper: "Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution"(http://vllab1.ucmerced.edu/~wlai24/LapSRN/) in PyTorch

## Usage
### Training
```
usage: main.py [-h] [--batchSize BATCHSIZE] [--nEpochs NEPOCHS] [--lr LR]
               [--step STEP] [--cuda] [--resume RESUME]
               [--start-epoch START_EPOCH] [--threads THREADS]
               [--momentum MOMENTUM] [--weight-decay WEIGHT_DECAY]
               [--pretrained PRETRAINED]

PyTorch LapSRN

optional arguments:
  -h, --help            show this help message and exit
  --batchSize BATCHSIZE
                        training batch size
  --nEpochs NEPOCHS     number of epochs to train for
  --lr LR               Learning Rate. Default=1e-4
  --step STEP           Sets the learning rate to the initial LR decayed by
                        momentum every n epochs, Default: n=10
  --cuda                Use cuda?
  --resume RESUME       Path to checkpoint (default: none)
  --start-epoch START_EPOCH
                        Manual epoch number (useful on restarts)
  --threads THREADS     Number of threads for data loader to use, Default: 1
  --momentum MOMENTUM   Momentum, Default: 0.9
  --weight-decay WEIGHT_DECAY, --wd WEIGHT_DECAY
                        weight decay, Default: 1e-4
  --pretrained PRETRAINED
                        path to pretrained model (default: none)

```

### Test
```
usage: test.py [-h] [--cuda] [--model MODEL] [--image IMAGE] [--scale SCALE]

PyTorch LapSRN Test

optional arguments:
  -h, --help     show this help message and exit
  --cuda         use cuda?
  --model MODEL  model path
  --image IMAGE  image name
  --scale SCALE  scale factor, Default: 4
```
We convert Set5 test set images to mat format using Matlab, for best PSNR performance, please use Matlab

### Prepare Training dataset
  - We provide a simple hdf5 format training sample in data folder with 'data', 'label_x2', and 'label_x4' keys, the training data is generated with Matlab Bicubic Interplotation, please refer [Code for Data Generation](https://github.com/twtygqyy/pytorch-vdsr/tree/master/data) for creating training files.

### Performance
  - We provide a pretrained LapSRN x4 model trained on T91 and BSDS200 images from [SR_training_datasets](http://vllab1.ucmerced.edu/~wlai24/LapSRN/results/SR_training_datasets.zip) with data augmentation as mentioned in the paper
  - No bias is used in this implementation, and another difference from paper is that Adam optimizer with 1e-4 learning is applied instead of SGD
  - Performance in PSNR on Set5, Set14, and BSD100
  
| DataSet/Method        | LapSRN Paper          | LapSRN PyTorch|
| ------------- |:-------------:| -----:|
| Set5      | 37.54      | **37.65** |
| Set14     | 28.19      | **28.27** |
| BSD100    | 27.32      | **27.36** |

### ToDos
  - LapSRN x8
  - Code for data generation 
  - LapGAN
  
### Citation

If you find the code and datasets useful in your research, please cite:
    
    @inproceedings{LapSRN,
        author    = {Lai, Wei-Sheng and Huang, Jia-Bin and Ahuja, Narendra and Yang, Ming-Hsuan}, 
        title     = {Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution}, 
        booktitle = {IEEE Conferene on Computer Vision and Pattern Recognition},
        year      = {2017}
    }