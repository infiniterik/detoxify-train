# Building docker files

`docker/Dockerfile.base` contains a multistage build container.
The first stage builds a bare-bones machine which can be provisioned for
data processing, training or testing.

# Data processing

Create the data processing image.
```
docker build --target data -t ihmc-bose/data .
```

# Citations

```bibtex
@inproceedings{bose-etal-2023-detoxifying,
    title = "Detoxifying Online Discourse: A Guided Response Generation Approach for Reducing Toxicity in User-Generated Text",
    author = "Bose, Ritwik  and Perera, Ian  and Dorr, Bonnie",
    booktitle = "Proceedings of the First Workshop on Social Influence in Conversations (SICon 2023)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.sicon-1.2",
    pages = "9--14"
}
```
