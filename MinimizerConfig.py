from dataclasses import dataclass
import os.path


@dataclass
class MinimizerConfig:
    # Save path configuration. All will be created when Minimizer is instantiated.
    parent_dir = os.path.dirname(os.path.abspath(__file__)) # Directory this file is in
    log_dir = os.path.join(parent_dir, "logs",)
    channels_dir = os.path.join(parent_dir, "save","data","channels")
    vectors_dir = os.path.join(parent_dir, "save","data","vectors")
    snapshots_dir = os.path.join(parent_dir, "save","data","vectors","snapshots")
    minimizer_json = os.path.join(parent_dir, "save","minimizer.json") #path to minimizer.json
    channels_json = os.path.join(parent_dir, "save","data","channels.json") #path to save data
    # Algorithm configuration
    deque_size : int = 20 # nr of iterations to keep track of to see improvements of entropy
    tolerance : float = 1e-15
    max_iterations : int = 100000
    # MinimizerModule configuration
    parallel_computations : int = 1
    epsilon : float = 1/1000
    # Saving configuration
    snapshot_interval : int = 5 # after how many iterations to save the current best vector

    # Output configuration
    verbose : bool = False
    log : bool = False
