from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse


class TextGenerator:
    def __init__(self, model_name, model_path, max_length=1000):
        self.max_length = max_length
        self.model_name = model_name
        self.bos_token = "<|beginning_of_sentence|>"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            bos_token=self.bos_token,
            eos_token="<|end_of_sentence|>",
            pad_token="<|pad|>",
        )
        self.model_path = model_path
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        self.model.eval()

    def generate(self, prompt, num_return_sequences=3):
        prompt = self.bos_token + prompt
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids

        generated_text_output = self.model.generate(
            input_ids=input_ids,
            max_length=self.max_length,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.9,
        )

        for i, output in enumerate(generated_text_output):
            print(f"{i+1}: {self.tokenizer.decode(output, skip_special_tokens=True)}")

    def generate_loop(self):
        while True:
            prompt = input("Prompt: ")
            self.generate(prompt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/opt-125m")
    parser.add_argument("--model_path", type=str)
    args = parser.parse_args()

    text_generator = TextGenerator(args.model_name, args.model_path)
    text_generator.generate_loop()
