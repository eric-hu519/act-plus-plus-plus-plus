import os
import glob

# 获取所有的h5py文件
h5py_files = sorted(glob.glob("/home/rlserver/Documents/Alohaljy/Aloha/act-plus-plus/datasets/sim_transfer_cube_scripted/episode_*.hdf5"))

# 对每个文件进行重命名
for i, file_path in enumerate(h5py_files):
    # 获取文件的目录和文件名
    dir_name = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)

    # 生成新的文件名
    new_base_name = "episode_" + str(i + 50) + ".hdf5"
    new_file_path = os.path.join(dir_name, new_base_name)

    # 重命名文件
    os.rename(file_path, new_file_path)