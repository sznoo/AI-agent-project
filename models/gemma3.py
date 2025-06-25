from transformers import AutoTokenizer, AutoModelForCausalLM


class Gemma3Wrapper:
    def __init__(self, model_id="google/gemma-3-12b-it"):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

    def infer(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        do_sample: bool = False,
        temperature: float = 0.1,
    ) -> str:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(
            self.model.device
        )
        output = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
