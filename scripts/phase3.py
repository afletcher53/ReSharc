from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

dataset = load_dataset("trl-lib/tldr", split="train")


# Define the reward function, which rewards completions that are close to 20 characters
def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]


training_args = GRPOConfig(
    output_dir="Qwen2-0.5B-GRPO",
    logging_steps=10,
    use_vllm=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    fp16=True,
    gradient_checkpointing=True,
    optim="adafactor",
)
trainer = GRPOTrainer(
    model="HuggingFaceTB/SmolLM-135M-Instruct",
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
