import os
import yaml

PARAMS_FILE = "params.yaml"

# DagsHub repo that holds your MLflow experiments
DAGSHUB_OWNER = "elango-ela"
DAGSHUB_REPO = "CICD-AWS"


def read_params(path: str = PARAMS_FILE) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def setup_mlflow(params: dict):
    """Log to DagsHub via the dagshub client; fall back to ./mlruns if it fails."""
    import mlflow  # lazy: not installed in the serving image

    if os.getenv("MLFLOW_TRACKING_URI"):
        # explicit override (e.g. token-based setup), nothing else needed
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    else:
        try:
            import dagshub
            dagshub.init(repo_owner=DAGSHUB_OWNER, repo_name=DAGSHUB_REPO, mlflow=True)
        except Exception as e:
            print("DagsHub setup failed, logging locally to ./mlruns:", e)

    print("MLflow tracking to:", mlflow.get_tracking_uri())
    mlflow.set_experiment(params["mlflow_experiment"])
    return mlflow