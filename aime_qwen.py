import torch
import torch.distributions as dist
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_ID = "Qwen/Qwen3-8B"  # Using Qwen2-7B as a readily available, strong model from the same family.
DATASET_ID = "HuggingFaceH4/aime_2024"
PROMPT_TEMPLATE = "User:\n{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n\nAssistant:\n"
TEMPERATURE = 1.0
MAX_NEW_TOKENS = 1_000_000  # Max tokens to generate for a single response.
OUTPUT_FILE = "aime24_response_entropies.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    dataset = load_dataset(DATASET_ID)

    responses_list = []
    for problem in dataset["train"]:
        question = problem["problem"]

        full_prompt = PROMPT_TEMPLATE.format(question=question)
        input_ids = tokenizer.encode(full_prompt, return_tensors="pt").to(DEVICE)
        prompt_length = input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                do_sample=True,
                temperature=TEMPERATURE,
                max_new_tokens=MAX_NEW_TOKENS,
                return_dict_in_generate=True,
            )

        generated_ids = outputs.sequences.squeeze()[prompt_length:]
        print(f"Number of Generated IDs: {len(generated_ids)}")
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(f"Generated Text: {generated_text}")

        with torch.no_grad():
            logits = model(outputs.sequences).logits.squeeze()[prompt_length:]

        entropies = dist.Categorical(logits=logits).entropy()
        response = {
            "id": problem["id"],
            "problem": question,
            "prompt": full_prompt,
            "solution": generated_text,
            "entropy": entropies.tolist(),
            "tokens": generated_ids.tolist(),
        }
        responses_list.append(response)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(responses_list, f, indent=4)
