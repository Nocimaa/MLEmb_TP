# Utiliser une image Python officielle légère
FROM python:3.11-slim

# Mettre à jour et installer les dépendances de base
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail
WORKDIR /app

# Installer MLflow
RUN pip install --upgrade pip
RUN pip install mlflow[extras]  # extras pour sqlite, postgres, s3 si besoin

# Exposer le port du serveur MLflow
EXPOSE 5000

# Créer les dossiers pour stocker les artefacts et la base de données
RUN mkdir -p /mlflow/artifacts /mlflow/db

# Définir les variables d'environnement pour MLflow
ENV MLFLOW_TRACKING_URI sqlite:///mlflow/db/mlflow.db
ENV MLFLOW_ARTIFACT_ROOT /mlflow/artifacts

# Commande pour démarrer le serveur MLflow
CMD mlflow server \
    --backend-store-uri $MLFLOW_TRACKING_URI \
    --default-artifact-root $MLFLOW_ARTIFACT_ROOT \
    --host 0.0.0.0 \
    --port 5000

