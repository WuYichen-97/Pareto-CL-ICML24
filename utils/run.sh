CUDA_VISIBLE_DEVICES=0 python3 main.py --model paretocl --dataset seq-cifar100 --lr 0.03 --n_epochs 1 --batch_size 32 --gamma 0.5 --buffer_size 1000 --minibatch_size 32
CUDA_VISIBLE_DEVICES=0 python3 main.py --model paretocl --dataset seq-cifar10 --lr 0.03 --n_epochs 1 --batch_size 32 --gamma 0.5 --buffer_size 1000 --minibatch_size 32
CUDA_VISIBLE_DEVICES=0 python3 main.py --model paretocl --dataset seq-tinyimg --lr 0.03 --n_epochs 1 --batch_size 32 --gamma 0.5 --buffer_size 2000 --minibatch_size 32
