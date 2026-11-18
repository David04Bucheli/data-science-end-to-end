FROM python:3.13-slim

WORKDIR /app

# dependencias primero, para que la cache sirva mientras no cambien
COPY servicio/requirements.txt servicio/requirements.txt
RUN pip install --no-cache-dir -r servicio/requirements.txt

COPY modelado/pipeline_proyecto.py modelado/pipeline_proyecto.py
COPY servicio/api.py servicio/api.py
COPY servicio/modelo.joblib servicio/modelo.joblib
COPY servicio/modelo_version.txt servicio/modelo_version.txt

EXPOSE 8000
CMD ["uvicorn", "servicio.api:app", "--host", "0.0.0.0", "--port", "8000"]