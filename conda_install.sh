eval "$(conda shell.bash hook)"

conda env remove -n epic_yolo
conda create -n epic_yolo python=3.6
conda activate epic_yolo

#pip install -r requirements.txt


# YOLOv5 requirements
# Usage: pip install -r requirements.txt

# Base ----------------------------------------
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install 'matplotlib>=3.2.2'
conda install 'numpy>=1.18.5'
conda install 'opencv-python>=4.1.1'
conda install 'Pillow>=7.1.2'
conda install 'PyYAML>=5.3.1'
conda install 'requests>=2.23.0'
conda install 'scipy>=1.4.1'
#conda install 'torch>=1.7.0'
#conda install 'torchvision>=0.8.1'
conda install 'tqdm>=4.41.0'
conda install 'protobuf<4.21.3'  # https://github.com/ultralytics/yolov5/issues/8012'

conda install 'tensorboard>=2.4.1'
conda install 'wandb'
conda install 'pandas>=1.1.4'
conda install 'seaborn>=0.11.0'
conda install ipython
conda install psutil  # system utilization'
conda install thop  # FLOPs computation'
