# Defense in Depth paper accompanying code

## Simple installation

If you just want to run the code and nothing else, you can do the following:

1. clone the repository
2. cd into it
3. create a new Python 3.10 virtual environment called `venv`
4. activate the virtual environment
5. install the `robust-llm` project

```
git clone https://github.com/AlignmentResearch/robust-llm.git
cd robust-llm
python -m venv venv
source venv/bin/activate
pip install .
```

Note that this project has not been tested with different versions of Python.

## Demo
Run `python demo.py` to query the defense pipeline interactively.