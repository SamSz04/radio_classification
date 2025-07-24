# Python script to chunk 10311×2×N frames into segments of length 512 or 1024

import os
import numpy as np
from scipy.io import loadmat, savemat

# 配置
base_path = r'D:\RadioData\20250711'   # 根目录
# distances = ['1m', '3m', '5m']         # 测距子目录
distances = ['1m']         # 测距子目录
mod_list = ['4ask','8ask','8psk','16psk',
            '16qam','32psk','32qam','64qam',
            '128qam','bpsk','ook','qpsk']

# 要切分的长度列表
segment_lengths = [64, 200]

for dist in distances:
    for mod in mod_list:
        mod_dir = os.path.join(base_path, dist, mod)
        if not os.path.isdir(mod_dir):
            continue

        for fn in os.listdir(mod_dir):
            if not fn.endswith('_frames.mat'):
                continue
            mat_path = os.path.join(mod_dir, fn)
            data = loadmat(mat_path)
            frames = data['frames']        # 原始 shape: (L, 2, N)
            Y      = data.get('Y', None)   # 如果有标签

            L, C, N = frames.shape
            print(f"Processing {fn}: length={L}, channels={C}, frames={N}")

            for seg_len in segment_lengths:
                nchunks = L // seg_len
                if nchunks == 0:
                    print(f"  Skipping seg_len={seg_len}: too long for L={L}")
                    continue

                total = nchunks * N
                # 新的数组预分配
                new_frames = np.zeros((seg_len, C, total), dtype=frames.dtype)
                new_Y = None
                if Y is not None:
                    # 重复标签
                    new_Y = np.repeat(Y.flatten(), nchunks)

                idx = 0
                # 分帧切片
                for i in range(N):
                    f = frames[:, :, i]  # shape (L,2)
                    for c in range(nchunks):
                        start = c * seg_len
                        new_frames[:, :, idx] = f[start:start+seg_len, :]
                        idx += 1

                # 构造输出文件名
                out_fn = fn.replace('_frames.mat', f'_seg{seg_len}.mat')
                out_path = os.path.join(mod_dir, out_fn)
                save_dict = {'frames': new_frames}
                if new_Y is not None:
                    save_dict['Y'] = new_Y
                savemat(out_path, save_dict)
                print(f"  Saved {out_fn}: new shape {new_frames.shape}")
