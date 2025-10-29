from contextlib import contextmanager
import mlflow

@contextmanager
def run_mlflow(run_name: str, params: dict | None = None, tags: dict | None = None):
    with mlflow.start_run(run_name=run_name):
        if params:
            mlflow.log_params(params)
        if tags:
            mlflow.set_tags(tags)
        yield

def log_metrics(metrics: dict, step: int | None = None):
    mlflow.log_metrics(metrics, step=step)
