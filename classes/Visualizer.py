from classes.Config import VisualizerConfig
import os.path
import json
import tensorflow as tf

class Visualizer:
    def __init__(self, config:VisualizerConfig=VisualizerConfig()):
        # Load configuration
        self.config = config

    def load_minimizer(self, id:str):
        """
        Loads the info about minimizer from the database.
        """
        # Load the JSON database
        print(self.config.minimizer_json)
        if not os.path.isfile(self.config.minimizer_json):
            raise FileNotFoundError(f"Database not found at {self.config.minimizer_json}")        
        with open(self.config.minimizer_json, "r") as f:
            database = json.load(f)

        # Now loop throught the database. Prune everything that is not relevant to minimizer with id "id".
        print(f"Database found. Now loading minimizer with id {id}...")
        self.database = []
        for entry in database:
            if "minimizer_id" in entry and entry["minimizer_id"] == id:
                self.database.append(entry)
        self.minimizer = id
        # If no match is found
        if not self.database:
            raise KeyError(f"No minimizer found with  with id {id} in {self.config.minimizer_json}")

        return self

    def get_runs(self):
        return [{entry["run_id"]:entry["entropy"]} for entry in self.database]
 
    def get_snapshots(self, run:str):
        for el in self.database:
            if el["run_id"] == run:
                return el["snapshots"]
        return []

    def load_vector(self, run:str, snapshot:str):
        fp = os.path.join(self.config.snapshots_dir,run,snapshot+".tfrecord")
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Could not find saved tensor at {fp}!")
        tensor_serialized = tf.io.read_file(fp)
        return tf.io.parse_tensor(tensor_serialized, out_type=tf.complex128)  

