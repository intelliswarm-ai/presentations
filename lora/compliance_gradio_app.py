import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Load your fine-tuned LoRA model here
# Replace with your actual path or Hugging Face model repo
model_name = "path/to/your/LoRA-compliance-model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).eval()

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

def compliance_assistant(instruction, input_text):
    prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Output:\n"
    result = pipe(prompt, max_new_tokens=512, do_sample=False, temperature=0.2)
    return result[0]['generated_text'].split("### Output:\n")[-1].strip()

# Gradio UI
gr.Interface(
    fn=compliance_assistant,
    inputs=[
        gr.Textbox(label="Instruction", placeholder="e.g. Draft SAR, flag risk...", lines=2),
        gr.Textbox(label="Input Data", placeholder="Paste KYC, email, or case description...", lines=10)
    ],
    outputs=gr.Textbox(label="LLM Output", lines=10),
    title="🛡️ Compliance LLM Assistant",
    description="Run your fine-tuned compliance LoRA model. Tasks include SAR generation, policy Q&A, risk flagging, etc."
).launch()
