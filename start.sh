#!/bin/bash

cd "$(dirname "$0")"

echo "[INFO] Czyszczenie pamięci .."
sudo swapoff -a
sudo fallocate -l 1G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

echo "[INFO] Budowanie i uruchamianie kontenera..."
docker compose up --build -d 

echo "[INFO] Czekam na logi (CTRL+C aby zakończyć)..."
docker compose logs -f

