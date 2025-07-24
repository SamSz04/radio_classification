import os
import numpy as np
from scipy.io import loadmat, savemat

# 配置路径
input_base_path = r'D:\RadioData\dataset5'     # 原始数据根目录
output_base_path = r'D:\RadioData\dataset5'    # 目标保存路径
os.makedirs(output_base_path, exist_ok=True)

distances = ['1m']  # 子目录（如测距条件）
mod_list = ['4ask', '8ask', '8psk', '16psk',
            '16qam', '32psk', '32qam', '64qam',
            '128qam', 'bpsk', 'ook', 'qpsk']

segment_lengths = [512]  # 切分长度

for dist in distances:
    for mod in mod_list:
        expected_filename = f"{mod}_frames.mat"
        mod_dir = os.path.join(input_base_path, dist, mod)
        if not os.path.isdir(mod_dir):
            continue

        for fn in os.listdir(mod_dir):
            if fn != expected_filename:
                continue  # 只处理严格匹配的 {mod}_frames.mat 文件

            mat_path = os.path.join(mod_dir, fn)
            data = loadmat(mat_path)

            if 'frames' not in data:
                print(f"  Skipped (no 'frames'): {fn}")
                continue

            frames = data['frames']        # (L, 2, N)
            Y = data.get('Y', None)

            # ==== 全局功率归一化 ====
            power = np.mean(np.abs(frames) ** 2)
            if power > 0:
                norm_factor = np.sqrt(power)
                frames = frames / norm_factor
            print(f"Processing {fn}: shape={frames.shape}, normalized by {norm_factor:.4f}")
            # ==========================

            L, C, N = frames.shape      # C==2，代表两个天线
            print(f"Processing {fn}: shape={frames.shape}")

            for seg_len in segment_lengths:
                nchunks = L // seg_len
                if nchunks == 0:
                    print(f"  Skipping seg_len={seg_len}: too long for L={L}")
                    continue

                total = nchunks * N
                new_frames = np.zeros((seg_len, C, total), dtype=frames.dtype)
                new_Y = None
                if Y is not None:
                    new_Y = np.repeat(Y.flatten(), nchunks)

                idx = 0
                for i in range(N):
                    f = frames[:, :, i]  # shape (L,2)
                    for c in range(nchunks):
                        start = c * seg_len
                        new_frames[:, :, idx] = f[start:start+seg_len, :]
                        idx += 1

                # 保存路径
                out_filename = f"{mod}_seg{seg_len}.mat"
                out_path = os.path.join(output_base_path, out_filename)

                save_dict = {'frames': new_frames}
                if new_Y is not None:
                    save_dict['Y'] = new_Y

                savemat(out_path, save_dict)
                print(f"  Saved: {out_filename} -> shape={new_frames.shape}")
