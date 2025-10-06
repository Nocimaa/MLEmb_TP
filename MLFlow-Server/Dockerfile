# Utiliser une image Python officielle légère
FROM python:3.11-slim

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail
WORKDIR /app

# Installer MLflow et ses extras (SQLite, S3, etc.)
RUN pip install --upgrade pip
RUN pip install mlflow[extras]

# Exposer le port de MLflow
EXPOSE 5000

# Créer les dossiers pour la DB et les artefacts avec permissions
RUN mkdir -p /mlflow/db /mlflow/artifacts \
    && chmod -R 777 /mlflow

# Variables d'environnement pour MLflow
ENV MLFLOW_TRACKING_URI sqlite:////mlflow/db/mlflow.db
ENV MLFLOW_ARTIFACT_ROOT /mlflow/artifacts

# Commande pour démarrer le serveur MLflow
CMD mlflow server \
    --backend-store-uri $MLFLOW_TRACKING_URI \
    --default-artifact-root $MLFLOW_ARTIFACT_ROOT \
    --host 0.0.0.0 \
    --port 5000
