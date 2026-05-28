# Create and activate the new environment as per the environment.yml file
mamba env create -f environment.yml

mamba activate abstention-bench

# Install VLLM and PyTorch using pip, because we need specific CUDA-compatible versions
pip install vllm==0.6.4.post1
 
if [[ $OSTYPE == "darwin"* ]]; then
  pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 -U 
else
  pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 -U --index-url https://download.pytorch.org/whl/cu121
fi 

pip install -e .

# Test that PyTorch is installed correctly
pytorch_version_output=`python -c "import torch; print(torch.__version__)"`
if [[ $pytorch_version_output == *"121"* ]]; then
  echo "PyTorch is installed with cuda 12.1!"
else
  echo "PyTorch installation missing cuda 12.1. Please install with pip using official instructions"
fi
