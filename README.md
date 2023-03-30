## Build a dataset for training
Example:
```python
python -c 'from models import data; data.make_prochoice_enrichment()'
```


## Convert model using Optimum

```python
python -m transformers.onnx --model=./my_model --feature=causal-lm-with-past --atol=1e-3 --opset 03 onnx/
```