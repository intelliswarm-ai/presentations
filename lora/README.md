# lora example for compliance

1) dataset 
compliance_lora_training_data.jsonl 

2) lora training with llama factory or PEFT
config.yaml for Llama Factory
https://github.com/hiyouga/LLaMA-Factory


3) Serve the model in a UI 
pip install gradio transformers torch
python compliance_gradio_app.py
