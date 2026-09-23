.PHONY: install train serve test drift docker-train docker-up

install:
	pip install -e .

train:
	python -m src.pipeline.train

serve:
	uvicorn src.api:app --host 0.0.0.0 --port 8000

test:
	pytest -q

drift:
	python -m src.monitoring

docker-train:
	docker compose run --rm train

docker-up:
	docker compose up api mlflow
