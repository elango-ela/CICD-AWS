# Brain Tumor MRI Classification: DVC + MLflow + Docker + GitHub Actions

Structure
```
pipeline/   stage_01 ingestion -> 02 preprocessing -> 03 training -> 04 evaluation, prediction.py
model/      model.h5 used by the app (commit it)
app.py      Flask UI + /predict      main.py   runs all stages
dvc.yaml    pipeline definition      params.yaml  all settings
```

## Run the pipeline locally
```bash
pip install -r requirements.txt
# 1. set data_url in params.yaml (already filled with your zip link)
# 2. (optional) MLflow on DagsHub, otherwise logs go to ./mlruns
export MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow
export MLFLOW_TRACKING_USERNAME=<user>
export MLFLOW_TRACKING_PASSWORD=<token>      # never commit this

git init && dvc init          # first time only
dvc repro                     # or: python main.py
dvc dag
mlflow ui                     # only if you did not set the DagsHub URI
```

## Deploy
```bash
git add model/model.h5 scores.json dvc.yaml dvc.lock params.yaml .dvc
git commit -m "trained model" && git push origin main
```
GitHub secrets needed: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `ECR_REPOSITORY_NAME`.
The EC2 instance must be registered as a self-hosted runner and have port 8080 open.

Local test: `docker build -t cnncls . && docker run -p 8080:8080 cnncls`

## Dataset
`params.yaml -> data_url` is one Google Drive zip (shared "Anyone with the link") containing two folders:
`yes/` (tumor) and `no/` (no tumor). Folder name = label (no=0, yes=1). Split: 90% train/validation, 10% test.
If the Drive download is blocked by quota, download the zip in the browser and unzip it so that
`artifacts/data_ingestion/dataset/yes` and `.../no` exist; ingestion will then skip the download.
