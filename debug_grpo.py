from datasets import load_dataset
from trl import GRPOTrainer
from trl.rewards import accuracy_reward

dataset = load_dataset("trl-lib/DeepMath-103K", split="train")
print(dataset.column_names)
print(dataset[0])

trainer = GRPOTrainer(
    model="./models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775",
    reward_funcs=accuracy_reward,
    train_dataset=dataset
)

trainer.train()
