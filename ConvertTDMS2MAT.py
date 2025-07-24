from nptdms import TdmsFile, TdmsWriter, ChannelObject
from scipy.io import savemat
import numpy as np
import os
import glob

dir_list = ["D:\\RadioData\\20250715_MS"]

for dirname in dir_list:
    for filename in glob.glob(f"{dirname}\\*.tdms"):
        if filename.endswith("_modified.tdms") or filename.endswith("_modified_modified.tdms"):
            continue

        print(f"正在处理：{filename}")
        base_name = os.path.basename(filename)
        name_wo_ext = os.path.splitext(base_name)[0]

        # ① 读取原始 TDMS 文件
        with TdmsFile.open(filename) as tdms_file:
            group = tdms_file['未命名']
            all_channels = list(group.channels())
            channel_map = {ch.name: ch[:] for ch in all_channels}

            print("该文件通道列表：", list(channel_map.keys()))

            # 尝试识别可用的前4个通道
            expected_names = ['Untitled', 'Untitled 1', 'Untitled 2', 'Untitled 3']
            channel_data = []

            for ch_name in expected_names:
                if ch_name in channel_map:
                    channel_data.append(channel_map[ch_name])
                else:
                    print(f"警告：找不到通道 {ch_name}，将用 0 填充")
                    # 用全 0 替代（保持一致长度）
                    if channel_data:
                        fill_len = len(channel_data[0])
                    else:
                        fill_len = 1000  # 兜底长度
                    channel_data.append([0.0] * fill_len)

            min_len = min(len(c) for c in channel_data)
            channel_data = [c[:min_len] for c in channel_data]

        # ② 保存为 _modified.tdms 文件
        mod_path = os.path.join(dirname, name_wo_ext + '_modified.tdms')
        with TdmsWriter(mod_path) as writer:
            ch_names = ['CH0-I', 'CH0-Q', 'CH1-I', 'CH1-Q']
            channels = [ChannelObject('dataflow', ch_names[i], channel_data[i]) for i in range(4)]
            writer.write_segment(channels)

        # ③ 可选转换：生成 _V2.tdms（模拟 tdmsconvert）
        v2_path = os.path.join(dirname, name_wo_ext + '_V2.tdms')
        with TdmsWriter(v2_path) as writer:
            writer.write_segment(channels)

        # ④ 读取 _V2.tdms 并导出为 .mat
        with TdmsFile.open(v2_path) as tdms_file:
            group = tdms_file['dataflow']
            dataflow = np.array([
                group['CH0-I'][:min_len],
                group['CH0-Q'][:min_len],
                group['CH1-I'][:min_len],
                group['CH1-Q'][:min_len]
            ])

        # 转置为 (n, 4)
        dataflow = dataflow.T

        # ⑤ 保存为 .mat 文件
        mat_path = os.path.join(dirname, name_wo_ext + ".mat")
        print(f"保存到：{mat_path}")
        if dataflow.nbytes > 2 * 1024 ** 3:
            import h5py
            with h5py.File(mat_path, 'w') as f:
                f.create_dataset("dataflow", data=dataflow, compression="gzip")
        else:
            savemat(mat_path, {"dataflow": dataflow})
