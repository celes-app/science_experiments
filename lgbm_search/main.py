import time
import numpy as np
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from IPython import embed
from src.logger import MLflowLogger
from src.data_loader import DataLoader
from src.config_loader import ConfigLoader
from src.hyperparameter_tunner import HyperparameterOptimizer
from darts.models import LightGBMModel


def main(
    use_mlflow: bool = False,
    parallel: bool = False,
    save_hyperparams: bool = False,
    n_jobs: int = -1
):
    # 1. Cargar configuración
    lgbm_path_config = "models/lgbm/search_space.yml"
    lgbm_config = ConfigLoader.load_yaml(lgbm_path_config)
    search_space = lgbm_config["search_space"]
    fixed_params = lgbm_config["fixed_params"]

    # 2. Cargar datos
    data_path = "/home/ec2-user/celes-ml/Forecast_test/data/f_demand_sample (1).parquet"
    data_loader = DataLoader(data_path)
    data_loader.load_data()
    data_loader.to_week()
    train_dict, val_dict, test_dict = data_loader.prepare_data_for_forecasting()

    # 3. MLflow logger
    mlflow_logger = MLflowLogger("optimizacion_2025-03-21")

    # 4. Crear el optimizador
    lgbm_optimizer = HyperparameterOptimizer(
        model_class=LightGBMModel,
        search_space=search_space,
        logger=mlflow_logger,
        model_name="lgbm",
        use_mlflow=use_mlflow
    )

    start_time = time.time()

    results = []
    series_keys = list(train_dict.keys())

    if parallel:

        parallel_results = Parallel(
            n_jobs=n_jobs,
            verbose=10, prefer="processes",
            max_nbytes=None)(
            delayed(lgbm_optimizer.optimize_one_series)(
                key,
                train_dict,
                val_dict,
                fixed_params,
                n_trials=5,
                save_hyperparams=False
            )
            for key in series_keys
        )

        for key, res_tuple in zip(series_keys, parallel_results):
            results.append((key,) + res_tuple)

    else:

        for key in series_keys:
            res_tuple = lgbm_optimizer.optimize_one_series(
                key,
                train_dict,
                val_dict,
                fixed_params,
                n_trials=5,
                save_hyperparams=False
            )

            results.append((key,) + res_tuple)

    global_rmse = []
    global_mape = []
    global_smape = []

    for (key, best_value, best_params, best_mape_, best_smape_, best_rmse_) in results:

        global_rmse.append(best_rmse_)
        global_mape.append(best_mape_)
        global_smape.append(best_smape_)


        if save_hyperparams:

            lgbm_optimizer.save_hyperparams_for_series(
                best_params,
                keysupplychain=key[0],
                trial_number="best"
            )

    elapsed = time.time() - start_time
    print(f"\nTiempo total: {elapsed:.2f} seg.")
    print(f"Tiempo total: {elapsed / 60:.2f} min.")
    print(f"Tiempo total: {elapsed / 3600:.2f} horas.")
    print(f"Tiempo total: {elapsed / 3600 / 24:.2f} dias.\n")

    if global_rmse:
        mean_rmse  = float(np.nanmean(global_rmse))
        mean_mape  = float(np.nanmean(global_mape))
        mean_smape = float(np.nanmean(global_smape))
        print("Métricas globales:")
        print(f"   RMSE promedio:  {mean_rmse:.4f}")
        print(f"   MAPE promedio:  {mean_mape:.4f}")
        print(f"   SMAPE promedio: {mean_smape:.4f}")
    else:
        print("No se obtuvieron métricas (¿no hay series?).")

    embed()


if __name__ == "__main__":
    main(use_mlflow=True, parallel=True, save_hyperparams=True, n_jobs=-1)
