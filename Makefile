.PHONY: train serve evaluate optimize benchmark test docker-up lint

train:
	python -m training.trainer --config configs/model_config.yaml

serve:
	uvicorn serving.app:app --reload

evaluate:
	python -m evaluation.ci_gate --config configs/evaluation_config.yaml

optimize:
	python -m optimization.onnx_exporter --model models/latest.pt

benchmark:
	python -m optimization.benchmarker --model models/latest.onnx

test:
	pytest tests/ -v

docker-up:
	docker-compose -f infrastructure/docker/docker-compose.yml up --build

lint:
	ruff check . && mypy .
