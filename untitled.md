in terminal:
sudo apt install python3.10-venv
python3 -m venv venv
source venv/bin/activate
python3 -m ipykernel install --user --name=venv --display-name "Python (venv)"

pip install torch torchvision matplotlib tqdm lmdb torchviz