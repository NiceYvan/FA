### FastAdaBelief opitmizer: A variant of AdaBelief for strongly convex loss functions. It converges fast in strongly convex conditions.


 
This repo heavily depends on the official implementation of AdaBelief: https://github.com/juntang-zhuang/Adabelief-Optimizer



### Dependencies
python 3.7
pytorch 1.1.0
torchvision 0.3.0
jupyter notebook 

### Training and evaluation code

(1) train network with
CUDA_VISIBLE_DEVICES=0 python main.py --optim fastadabelief --lr 1e-3 --eps 1e-8 --beta1 0.9 --beta2 0.999 --momentum 0.9

--optim: name of optimizers 
--lr: learning rate
--eps: epsilon value used for optimizers. Note that Yogi uses a default of 1e-03, other optimizers typically uses 1e-08
--beta1, --beta2: beta values in adaptive optimizers
--momentum: momentum used for SGD.s

(2) visualize using the notebook "visualization.ipynb"


 
