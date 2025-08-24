# 🩺 Ocular Disease CNN

Clasificador de imágenes para detección de enfermedades oculares (Cataract, Diabetic Retinopathy, Glaucoma, Normal) basado en **PyTorch**.

Este proyecto incluye:
- Estructura de repo profesional (`src/`, `scripts/`, `data/`, `runs/`, `checkpoints/`).
- Entrenamiento reproducible con **PyTorch puro**.
- Visualización de métricas en **TensorBoard**.
- Guardado de **checkpoints** (`.pth`) para reproducir y desplegar modelos.
- Flujo completo con **Makefile** y **Docker** (CPU o GPU automático).

---

## 🚀 Instalación local (con [uv](https://docs.astral.sh/uv/))

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/tu-usuario/ocular-disease-cnn.git
   cd ocular-disease-cnn
   ```

2. Instalar dependencias:
   ```bash
   uv sync
   ```

3. Descargar el dataset de Kaggle:
   ```bash
   make download
   ```
   Esto dejará los datos en `data/dataset/` con subcarpetas por clase.

---

## 🏋️ Entrenamiento

Entrenar en local:
```bash
make train
```

Entrenar con hiperparámetros personalizados:
```bash
EPOCHS=5 BATCH_SIZE=64 LR=1e-4 make train
```

Los logs se guardan en `runs/exp1` (configurable con `RUNS_DIR`), y los checkpoints en `checkpoints/`.

---

## 📊 Visualización con TensorBoard

Lanzar servidor TensorBoard:
```bash
make tensorboard
```

Abrir en navegador: [http://localhost:6006](http://localhost:6006)

---

## ✅ Evaluación

Evaluar el mejor modelo (`checkpoints/best_model.pth`) en el dataset completo:
```bash
make eval
```

---

## 🔍 Predicción

Clasificar imágenes individuales:
```bash
make predict IMG="data/test1.jpg data/test2.jpg"
```

---

## 🐳 Uso con Docker

### Build de la imagen
```bash
make docker-build
```

### Entrenamiento en contenedor
```bash
make docker-train
```

### TensorBoard en contenedor
```bash
make docker-tensorboard
```

Volúmenes montados:
- `./data` → `/app/data`
- `./runs` → `/app/runs`
- `./checkpoints` → `/app/checkpoints`

---

## 📂 Estructura del proyecto

```
ocular-disease-cnn/
├── src/ocular_disease_cnn/
│   ├── model.py        # Definición CNN
│   ├── train.py        # Entrenamiento
│   ├── eval.py         # Evaluación best_model.pth
│   └── predict.py      # Inferencia en imágenes
├── scripts/
│   └── download_data.py
├── data/               # Dataset (ignorado en git)
├── runs/               # Logs TensorBoard (ignorado en git)
├── checkpoints/        # Modelos entrenados (ignorado en git)
├── Makefile
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

---

## ⚙️ Variables de entorno útiles

- `EPOCHS` → nº de épocas (default 10)
- `BATCH_SIZE` → tamaño de batch (default 32)
- `LR` → learning rate (default 1e-3)
- `IMG_SIZE` → tamaño de entrada (default 224)
- `VAL_RATIO` → proporción de validación (default 0.2 si hay `data/dataset`)
- `RUNS_DIR` → carpeta para logs (default `runs/exp1`)
- `CKPT_DIR` → carpeta para checkpoints (default `checkpoints/`)
- `BEST_METRIC` → métrica para elegir best_model (`val_acc` o `val_loss`)

Ejemplo:
```bash
EPOCHS=20 LR=5e-4 RUNS_DIR=runs/exp2 make train
```

---

## 🧰 Roadmap

- [x] Entrenamiento CNN básico
- [x] TensorBoard + barras de progreso
- [x] Checkpoints por época + best_model
- [x] Scripts de eval y predict
- [x] Docker universal (CPU/GPU)
- [ ] Añadir métricas extra (F1, Confusion Matrix)
- [ ] Exportar modelo a TorchScript/ONNX
- [ ] Integración con Runpod (multiusuario)
