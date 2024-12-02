# [Mitigating Catastrophic Forgetting in Online Continual Learning by Modeling Previous Task Interrelations via Pareto Optimization]([https://github.com/WuYichen-97/Meta-Continual-Learning-Revisited-ICLR2024](https://openreview.net/pdf?id=olbTrkWo1D)) 
ICML'24: Mitigating Catastrophic Forgetting in Online Continual Learning by Modeling Previous Task Interrelations via Pareto Optimization  (Official Pytorch implementation).  

If you find this code useful in your research then please cite  
```bash
@inproceedings{wumitigating,
  title={Mitigating Catastrophic Forgetting in Online Continual Learning by Modeling Previous Task Interrelations via Pareto Optimization},
  author={Wu, Yichen and Wang, Hong and Zhao, Peilin and Zheng, Yefeng and Wei, Ying and Huang, Long-Kai},
  booktitle={Forty-first International Conference on Machine Learning}
}
``` 



## Setups
The required environment is as follows:  

- Linux 
- Python 3+
- PyTorch/ Torchvision


## Running our method on benchmark datasets (CIFAR-10/100 & TinyImageNet).
Here is an example (utils/run.sh):
```bash
cd utils
CUDA_VISIBLE_DEVICES=0 python3 main.py --model paretocl --dataset seq-cifar100 --lr 0.03 --n_epochs 1 --batch_size 32 --gamma 0.5 --buffer_size 1000 --minibatch_size 32
CUDA_VISIBLE_DEVICES=0 python3 main.py --model paretocl --dataset seq-cifar10 --lr 0.03 --n_epochs 1 --batch_size 32 --gamma 0.5 --buffer_size 1000 --minibatch_size 32
CUDA_VISIBLE_DEVICES=0 python3 main.py --model paretocl --dataset seq-tinyimg --lr 0.03 --n_epochs 1 --batch_size 32 --gamma 0.5 --buffer_size 2000 --minibatch_size 3
```


## Acknowledgements
We thank the Pytorch Continual Learning framework *Mammoth*(https://github.com/aimagelab/mammoth)
