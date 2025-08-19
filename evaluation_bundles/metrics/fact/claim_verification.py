import argparse
import sys, os
import json
root = os.path.abspath(
    os.path.join(__file__, os.pardir, os.pardir, os.pardir, os.pardir)
)
models_dir = os.path.join(root, "models")

sys.path.insert(0, models_dir)

from generations_pipeline import LLMGenerator

class ClaimVerification(LLMGenerator):
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        cache_dir: str = "/import/nlp-datasets/LLMs/",
        **kwargs,
    ):
        
        self.prompt = '''
        You are a fact-checking assistant. Your task is to verify the truthfulness of a claim based on the provided text.
        You will receive an input that is structured as follows:
        Claim: <claim>
        Text: <text>
        Your task is to determine whether the text supports the claim or not.
        Only respond with "TRUE" or "FALSE". 
        Respond with "TRUE" if the claim is supported by the text, otherwise respond with "FALSE".
        '''
        self.generator = LLMGenerator(
            model_name=model_name
        )
    
    def verify(self, claim: str, text: str):
        user_text = f"Claim: {claim}\nText: {text}"
        return self.generator.generate(
            prompt=self.prompt,
            text=user_text,
            max_tokens=500,
            temperature=0.0
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Claim Verification using LLMs.")
    parser.add_argument("--test", type=bool, default=False, help="Run test predictions")
    parser.add_argument("--claims_file", type=str, help="Path to claims file")
    parser.add_argument("--text_file", type=str, help="Path to text file")
    args = parser.parse_args()

    with open(args.claims_file, "r") as f:
        claims = json.load(f)
    with open(args.text_file, "r") as f:
        texts = json.load(f)

    print("Loading model")
    verifier = ClaimVerification()
    all_results = {}
    for claim_id, claim_list in claims.items():
        text = texts[claim_id]
        print(f"Verifying claims: {text}")
        result = []
        for claim in claim_list:
            print(claim)
            result.append(verifier.verify(claim, text))

        all_results[claim_id] = result
    # Save the results to a file
    output_file = "claim_verification_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"Claim verification results saved to {output_file}.")

