install:
	uv pip install .[dev,test]

prepare:
	black .
	pytest tests