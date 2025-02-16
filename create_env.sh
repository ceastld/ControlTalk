ml CUDA/12.4.0
conda deactivate
conda env remove -n control -y
conda env create -f environment.yml
conda activate control
pip install psutil
pip install huggingface_hub
cd checkpoints
huggingface-cli download TencentGameMate/chinese-hubert-large --local-dir TencentGameMate/chinese-hubert-large --exclude "*.git*" "README.md" "docs"
huggingface-cli download Lavivis/ControlTalk --local-dir . --exclude "*.git*" "README.md" "docs"
cd ..
