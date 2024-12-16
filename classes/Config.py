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
    tolerance : float = 1e-15
    deque_size : int = 20
    max_iterations : int = 100000      
    entropy_to_track : int = 0 # 0: track the entropy epsilon; 1: track the estimated entropy; 2: track the upper bound on the estimated entropy; 3: track the lower bound on the estimated entropy.
    exponential_fit_window_size = 200 # Number of samples to use for exponential fit
    exponential_fit_Rsquared_min = 0.995 # How good the fit must be before using it to predict

    # MOE finding configuration
    MOE_attempts : int = 100 # How many times to run the algorithm to find MOE
    MOE_use_prediction : bool = True # Discard attempts that don't look promising based on MOE prediction
    MOE_prediction_tolerance : float = 1e-2 # How much bigger the predicted value needs to be, before discarding the attempt.

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

@dataclass
class VisualizerConfig(FolderConfig):
        show_sphere = True
