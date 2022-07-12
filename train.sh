eval "$(conda shell.bash hook)"

conda activate /home/sxa180157/anaconda3/envs/epic_yolo

python train.py --img 1280 --batch 32 --epochs 20 --data get_data/epic_kitchens.yaml --weights yolov5m.pt \
--cache disk
#makes it faster