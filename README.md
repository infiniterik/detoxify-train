# Building docker files

`docker/Dockerfile.base` contains a multistage build container.
The first stage builds a bare-bones machine which can be provisioned for
data processing, training or testing.

# Data processing

Create the data processing image.
```
docker build --target data -t ihmc-bose/data .
```


