import os
import numpy as np
from config import raw_data_path, univariate_data_path, processed_data_path, models_path

save_dir = os.path.join(univariate_data_path, "target_univariate.npy")
target_data = np.load(save_dir, allow_pickle=True)
target_data = [item for item in target_data if item['preterm'] is not None]


