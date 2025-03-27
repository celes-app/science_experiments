import yaml
import importlib

class ConfigLoader:

    @staticmethod
    def load_yaml(config_path: str):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    @staticmethod
    def load_metrics(metrics_config: list):
        metrics = {}
        for metric in metrics_config:
            module_name, func_name = metric['function'].rsplit('.', 1)
            module = importlib.import_module(module_name)
            metrics[metric['name']] = getattr(module, func_name)
        return metrics
