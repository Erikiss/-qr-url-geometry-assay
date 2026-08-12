.PHONY: install lint test smoke benchmark docker

install:
	python -m pip install -e '.[test]'

lint:
	ruff format --check .
	ruff check .

test:
	pytest

smoke:
	qr-assay demo-data --output-dir data/raw/demo --pairs 250
	qr-assay run --config configs/smoke.yml

benchmark:
	qr-assay benchmark --count 1000 --workers 0

docker:
	docker build -t qr-url-geometry-assay:latest .
