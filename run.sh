echo "Building dataset"
#python -c 'from models import data; data.make_prochoice_enrichment()'

echo "Training Model"
python models/neo_gpt.py configs/config-t5.json

echo "Converting to ONNX"
python -m transformers.onnx --model=./my_model --feature=causal-lm-with-past --atol=1e-3 --opset 03 onnx/

echo "Testing Model"
python models/test.py >> test.txt
