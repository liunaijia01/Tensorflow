import yaml
import os

cur_path = os.path.abspath(os.path.dirname(__file__))
config_path = os.path.join(cur_path, "../config/application_prod.yaml")

class ConfigHelper:
    def __init__(self):
        self.config_path = config_path
        self.config = self.load_config()


    def load_config(self):
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
        return config

    def get_value_from_config(self, key):
        subkeys = key.split(".")
        tmp = self.config
        for attr in subkeys:
            if attr not in tmp:
                return None
            tmp = tmp[attr]
        return tmp

    def get(self, key):
        return self.get_value_from_config(key)

config_helper = ConfigHelper()


if __name__ == '__main__':
    print(config_helper.get("hdfs"))
    print(config_helper.get("hdfs.ceshi"))