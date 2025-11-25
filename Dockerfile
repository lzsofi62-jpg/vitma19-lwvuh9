# 1. Base Image
FROM python:3.11-slim

# 2. Munkakönyvtár beállítása
WORKDIR /app

# 3. Rendszerszintű függőségek telepítése
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 4. Python függőségek másolása és telepítése
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Alkalmazás kódjának másolása
COPY ./src .

# 6. Script futtathatóvá tétele
RUN chmod +x run.sh

# 7. Konténer indításkor a pipeline fut
CMD ["bash", "run.sh"]
