# Utiliser une image Python légère
FROM python:3.11-slim

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Installer MLflow et extras pour sklearn
RUN pip install --no-cache-dir mlflow[extras] scikit-learn

# Définir le répertoire de travail
WORKDIR /app

# Créer les dossiers pour la base de données et les artifacts
RUN mkdir -p /app/mlflow-artifacts

# Variables d'environnement MLflow
ENV MLFLOW_TRACKING_URI=sqlite:///app/mlflow.db
ENV MLFLOW_ARTIFACT_ROOT=/app/mlflow-artifacts

# Lancer le serveur MLflow
CMD mlflow server \
    --backend-store-uri $MLFLOW_TRACKING_URI \
    --default-artifact-root $MLFLOW_ARTIFACT_ROOT \
    --host 0.0.0.0 \
    --port 5000
