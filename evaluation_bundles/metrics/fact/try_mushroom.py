import argparse
from mushroom import Mushroom
import json

parse = argparse.ArgumentParser(description="Decompose clinical notes into claims.")
parse.add_argument("--input_file", type=str, help="Path to the input JSON file containing clinical notes.")

args = parse.parse_args()

with open(args.input_file, "r") as f:
    data = json.load(f)

mush = Mushroom(max_claim=10)

references = []
ids = []

for key, value in data.items():
    references.append(value["reference"])
    ids.append(key)

decomposed_references = mush.decompose(texts=references, batch_size=12, max_new_tokens=2000)

claims_by_id = {}
for i, ref in enumerate(decomposed_references):
    # Only take assistant response
    claims_full = ref.split("\n\nassistant\n<think>\n\n")[1].strip()
    # split by "\nClaim" to get individual claims
    claim_parts = claims_full.split("\nClaim ")
    # Split by claims, each claim starts with "Claim"
    claims = []
    for claim in claim_parts:
        claim = claim.split(":", 1)[-1].strip()
        claims.append(claim)
    claims_by_id[ids[i]] = claims
    

# Save the claims to a file
output_file = "output_decomposed_claims.json"
with open(output_file, "w") as f:
    json.dump(claims_by_id, f, indent=4)
print(f"Decomposed claims saved to {output_file}.")
