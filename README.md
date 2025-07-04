# local-rag

git clone https://github.com/rathijit4u/local-rag.git

cd to project root directory

Run following commands -

For Windows - 
.\local-rag\Scripts\Activate.ps1
Check cuda version - 
nvcc --version

install pytorch for specific cuda version
https://pytorch.org/get-started/locally/

python.exe -m pip install --upgrade pip

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

pip install -r requirement.txt

python -m spacy download en_core_web_trf





