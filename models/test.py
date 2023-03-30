from neo_gpt import load_model, CausalLMDataset
from datasets import load_from_disk


tokenizer, model = load_model("./onnx/", "./onnx/", use_onnx=True)

ds = load_from_disk("data/prochoice.enriched.toxicity")
cds = CausalLMDataset.create(ds, None, max_length=512)

from tqdm import tqdm

cds["enrichment_test"].return_text = True
generate = lambda d: model(d, 
                            max_new_tokens=512, 
                            pad_token_id=tokenizer.pad_token_id,
                            do_sample=True,
                            top_k=50,
                            top_p=0.95,
                            num_return_sequences=1,
                            temperature=0.9,)
for d in tqdm(cds["enrichment_test"]):
    nontoxic = generate(d)
    toxic = generate(d.replace("A non-toxic reply:", "A toxic reply:"))
    
    res1 = nontoxic[0]['generated_text'].split("A non-toxic reply: ")[1]
    res2 = toxic[0]['generated_text'].split("A toxic reply: ")[1]
    if res1 != res2:
        print(d)
        print("nontoxic:\t", res1)
        print("toxic:\t", res2)
        print("----")