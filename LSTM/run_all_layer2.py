import os

# 2-layer lstm
cmd = ' python main.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --epoch 200 --save PTB.pt --when 100 145 --clip 0.25 --beta1 0.9 --beta2 0.9 --optimizer adabelief --lr 0.01 --eps 1e-12 --eps_sqrt 0.0 --nlayer 2 --run 0'
os.system(cmd)


cmd = ' python main.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --epoch 200 --save PTB.pt --when 100 145 --clip 0.25 --beta1 0.9 --beta2 0.999 --optimizer sgd --lr 30 --eps 1e-8 --nlayer 2 --run 0'
os.system(cmd)