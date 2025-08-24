# Variables
PYTHON = uv run python
PKG = ocular_disease_cnn

# ========================
# Targets principales
# ========================

# Descargar dataset Kaggle y dejarlo en data/dataset/
download:
	$(PYTHON) scripts/download_data.py

# Entrenar modelo (usa dataset completo y split interno)
train:
	$(PYTHON) -m $(PKG).train

# TensorBoard para visualizar resultados
tensorboard:
	uv run tensorboard --logdir runs

# Limpieza de logs y resultados
clean:
	rm -rf runs/*

# Limpieza total de datos + logs (cuidado)
clean-all: clean
	rm -rf data/dataset data/train data/val

# Evaluar mejor modelo guardado
eval:
	$(PYTHON) -m $(PKG).eval

# Predecir sobre imágenes (ejemplo: make predict IMG="path/to/img1.jpg path/to/img2.jpg")
predict:
	$(PYTHON) -m $(PKG).predict $(IMG)

# ========================
# Docker targets
# ========================

# Build de la imagen Docker
docker-build:
	docker compose build

# Entrenamiento dentro de Docker
docker-train:
	docker compose up

# TensorBoard dentro de Docker (con puerto 6006 expuesto)
docker-tensorboard:
	docker compose run --service-ports ocular-cnn make tensorboard

# Evaluación dentro de Docker
docker-eval:
	docker compose run ocular-cnn make eval

# Predicción dentro de Docker (con imagen específica)
docker-predict:
	docker compose run ocular-cnn make predict IMG=$(IMG)