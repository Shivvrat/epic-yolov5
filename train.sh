eval "$(conda shell.bash hook)"

conda activate /home/sxa180157/anaconda3/envs/epic_yolo

python train.py --img 640 --batch 16 --epochs 50 --data get_data/epic_kitchens.yaml --weights yolov5l.pt \
--cache ram
#makes it faster