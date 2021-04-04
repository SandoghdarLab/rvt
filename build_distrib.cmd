del /Q build
del /Q dist
python -m build
@REM twine upload dist/*
twine upload --repository testpypi dist/*
pip install --upgrade --index-url https://testpypi.python.org/pypi imgrvt