python -c 'from models import data; data.make_prochoice_enrichment()'

python models/neo_gpt.py

python -m transformers.onnx --model=./my_model --feature=causal-lm-with-past --atol=1e-3 --opset 03 onnx/

python models/test.py >> test.txt