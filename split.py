import os
import numpy as np
import shutil
from tqdm import tqdm

origin_path = 'data/humanscan_wbg'
sample_list = os.listdir(os.path.join(origin_path, 'human_train'))
num_samples = len(sample_list)
val_index = np.random.choice(num_samples, 26, replace=False)
val_sample = [sample_list[i] for i in val_index]
for item in tqdm(val_sample[26:]):
    shutil.move(os.path.join(origin_path, 'human_train', item), os.path.join(origin_path, 'human_val', item))

for item in tqdm(val_sample[:26]):
    shutil.move(os.path.join(origin_path, 'human_train', item), os.path.join(origin_path, 'human_test', item))

