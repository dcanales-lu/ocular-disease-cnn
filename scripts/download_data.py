from pathlib import Path
import shutil
import kagglehub

# Descarga (usa caché si ya lo tienes)
src_root = Path(kagglehub.dataset_download("gunavenkatdoddi/eye-diseases-classification"))
src_dataset = src_root / "dataset"  # contiene cataract/, diabetic_retinopathy/, glaucoma/, normal/

# Destino dentro del proyecto
dst_dataset = Path("data/dataset")

# Limpia y copia el dataset completo (sin splits)
if dst_dataset.exists():
    shutil.rmtree(dst_dataset)
shutil.copytree(src_dataset, dst_dataset)

print(f"Dataset preparado en: {dst_dataset.resolve()}")
print("Estructura esperada: data/dataset/{cataract,diabetic_retinopathy,glaucoma,normal}")
