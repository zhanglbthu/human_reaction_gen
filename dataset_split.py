import os
import glob
# 根目录
root_dir = "./Data/VIMO"
video_dir = "./Data/VIMO/cam_align"
category = "explosion"

# 输入文件
splits = {
    "train": "./Data/VIMO/backup/train.txt",
    "test": "./Data/VIMO/backup/test.txt"
}

# 输出目录
out_dir = os.path.join(root_dir, category)
os.makedirs(out_dir, exist_ok=True)

for split_name, split_file in splits.items():
    new_lines = []
    with open(split_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            video_path, npy_path = line.split()
            # 判断是否是目标类别
            if f"videos/{category}/" not in video_path:
                continue

            # 提取视频文件名，例如 explosion-007
            base_name = os.path.splitext(os.path.basename(video_path))[0]

            # 检查 root_dir/category/base_name/gen.mp4
            # gen_path = os.path.join(video_dir, category, base_name, "gen.mp4")
            pattern = os.path.join(video_dir, category, base_name, "*", "gen.mp4")
            matches = glob.glob(pattern)
            if matches:
                gen_path = matches[0]
            else:
                gen_path = None
            if gen_path is not None:
                # 替换原来的 video_path
                gen_path = os.path.relpath(gen_path, root_dir)
                new_line = f"{gen_path} {npy_path}"
                new_lines.append(new_line)

    # 保存新文件
    out_file = os.path.join(out_dir, f"{split_name}.txt")
    with open(out_file, "w") as f:
        f.write("\n".join(new_lines))

    print(f"Saved {len(new_lines)} lines to {out_file}")
