set -e

pip install --upgrade git+https://github.com/knoxml/simplet5

mkdir -p outputs/
python models/t5.py -m "test()"
