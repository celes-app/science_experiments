import mlflow

class MLflowLogger:

    def __init__(self, experiment_name: str):
        mlflow.set_experiment(experiment_name)

    def log_trial(self, params: dict, metrics: dict, tags: dict = None, run_name: str = None):
        """
        Registra un 'trial' en MLflow: parámetros, métricas, y tags opcionales.
        Utiliza run_name si deseas un nombre concreto; si no se provee, se generará uno.
        """
        # Si ya hay un run activo, iniciamos uno anidado; si no, iniciamos uno normal.
        if mlflow.active_run() is not None:
            run_ctx = mlflow.start_run(nested=True, run_name=run_name)
        else:
            run_ctx = mlflow.start_run(run_name=run_name)

        with run_ctx:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            if tags:
                for k, v in tags.items():
                    mlflow.set_tag(k, v)

    def log_artifact(self, artifact_path: str):
        """
        Permite registrar un archivo local como artefacto en MLflow, por ejemplo un CSV de métricas.
        """
        mlflow.log_artifact(artifact_path)
