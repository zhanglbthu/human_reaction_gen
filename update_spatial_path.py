import os

def update_test_file(test_file, videos_new_root, output_file):
    # Read all lines
    with open(test_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    updated_lines = []

    for line in lines:
        video_path, label_path, depth_path = line.split()[0], line.split()[1], line.split()[2]
        # Extract video type and sequence name, e.g. automobile_rush_towards, automobile_rush_towards-001
        parts = video_path.split('/')
        if len(parts) < 3:
            print(f"⚠️ Skip invalid line: {line}")
            continue
        elif len(parts) == 3:
            video_type = parts[1]                # e.g. automobile_rush_towards
            video_seq = parts[2].replace(".mp4", "")  # e.g. automobile_rush_towards-001
        elif len(parts) == 6:
            video_type = parts[2]                # e.g. automobile_rush_towards
            video_seq = parts[3].replace(".mp4", "")  # e.g. automobile_rush_towards-001
        else:
            print(f"⚠️ Skip invalid line: {line}")
            continue

        # Construct expected directory in videos_new
        search_dir = os.path.join(videos_new_root, video_type)
        
        # Find gen.mp4 recursively in that directory
        new_video_path = None
        if os.path.exists(search_dir):
            for root, _, files in os.walk(search_dir):
                if "gen.mp4" in files:
                    rel_path = os.path.relpath(os.path.join(root, "gen.mp4"), videos_new_root)
                    new_video_path = f"videos_new/expert/{rel_path}".replace("\\", "/")
                    break

        # If found, replace the path
        if new_video_path:
            updated_line = f"{new_video_path} {label_path} {depth_path}"
        else:
            # Keep the original if not found
            updated_line = line
            # print(f"⚠️ gen.mp4 not found for: {video_seq}")

        updated_lines.append(updated_line)

    # Write to new file
    with open(output_file, "w") as f:
        f.write("\n".join(updated_lines))

    print(f"✅ Updated test file saved to: {output_file}")

if __name__ == "__main__":
    # Example usage
    root_dir = "Data/VIMO"
    # test_file = "test.txt"
    # videos_new_root = "videos_new"
    # output_file = "test_updated.txt"
    test_file = os.path.join(root_dir, "test_spatial.txt")
    videos_new_root = os.path.join(root_dir, "videos_new", "expert")
    output_file = os.path.join(root_dir, "test_new.txt")

    update_test_file(test_file, videos_new_root, output_file)