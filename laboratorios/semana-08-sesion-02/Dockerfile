FROM python:3.12-slim

WORKDIR /app

#lo copio a la carpeta de la raíz
COPY requirements.txt . 

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "./starter/EDA.py"]