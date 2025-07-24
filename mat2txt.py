import os
import scipy.io
import numpy as np


def convert_mat_to_txt(mat_file, txt_file):
    # 加载.mat文件
    mat_data = scipy.io.loadmat(mat_file)

    # 获取.mat文件中的所有变量名
    variable_names = [var for var in mat_data if not var.startswith('__')]  # 排除MATLAB的内建字段

    # 打开.txt文件写入数据
    with open(txt_file, 'w') as f:
        for var_name in variable_names:
            data = mat_data[var_name]

            # 检查数据类型，如果是复数信号，拆分实部和虚部
            if np.iscomplexobj(data):
                real_part = np.real(data)
                imag_part = np.imag(data)
            else:
                # 如果是实数信号，虚部设置为0
                real_part = data
                imag_part = np.zeros_like(data)

            # 写入当前变量的数据，去除方括号
            for r, i in zip(real_part.flatten(), imag_part.flatten()):
                f.write(f"{r},{i}\n")

    print(f"Data has been successfully written to {txt_file}.")


def process_all_mat_files_in_folder(input_folder, output_folder):
    # 获取文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 只处理.mat文件
        if filename.endswith('.mat'):
            mat_file = os.path.join(input_folder, filename)
            txt_file = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.txt")
            convert_mat_to_txt(mat_file, txt_file)


# 示例使用
input_folder = 'D:\RadioData\ModulationSignal'  # 输入文件夹路径
output_folder = 'D:\RadioData\ModulationSignal'  # 输出文件夹路径

process_all_mat_files_in_folder(input_folder, output_folder)
