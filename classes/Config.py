from dataclasses import dataclass
import os.path

@dataclass 
class FolderConfig:
    # Save path configuration. All will be created when Minimizer is instantiated.
    parent_dir : str = os.path.dirname(os.path.abspath(__file__)) # Parent directory to create "save" folder in.
    # All these are now relative paths.
    log_dir : str = os.path.join("logs",)
    channels_dir : str = os.path.join("save","data","channels")
    vectors_dir : str = os.path.join("save","data","vectors")
    snapshots_dir : str = os.path.join("save","data","vectors","snapshots")
    minimizer_json : str = os.path.join("save","minimizer.json") #path to minimizer.json
    channels_json : str = os.path.join("save","data","channels.json") #path to save data

    def __post_init__(self):
        # Turn relative paths into absolute paths
        self.log_dir : str = os.path.join(self.parent_dir, self.log_dir)
        self.channels_dir : str = os.path.join(self.parent_dir, self.channels_dir)
        self.vectors_dir : str = os.path.join(self.parent_dir, self.vectors_dir)
        self.snapshots_dir : str = os.path.join(self.parent_dir, self.snapshots_dir)
        self.minimizer_json : str = os.path.join(self.parent_dir, self.minimizer_json)
        self.channels_json : str = os.path.join(self.parent_dir, self.channels_json)
     
        

@dataclass
class MinimizerConfig(FolderConfig):
    # Algorithm configuration
    deque_size : int = 20 # nr of iterations to keep track of to see improvements of entropy
    tolerance : float = 1e-15
    max_iterations : int = 100000
    # MinimizerModule configuration
    parallel_computations : int = 1
    epsilon : float = 1/10000
    # Saving configuration
    save : bool = False # If false, no snapshot is taken and the database is not updated.
    snapshot_interval : int = 20 # after how many iterations to save the current best vector

    # Output configuration
    verbose : bool = False
    log : bool = False
    log_level : int = 1 #0 only prints essential messages like start and end of run info. 1 prints all messages.
    log_entropy : int = 0 # 0, log the epsilon entropy. 1, log the est entropy. 2, log the ub on est entropy, 3, log the lb on estimated entropy.

@dataclass
class VisualizerConfig(FolderConfig):
        show_sphere = True
