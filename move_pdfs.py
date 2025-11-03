import os, shutil

# source and destination folders
src_root = r"C:\Users\mckin\OneDrive\Desktop\syncthing-folder\Git-Repos\wellfound-bot\medicare\UnitedHealthcare\uhc_plan_pdfs"
dst_root = r"C:\Users\mckin\OneDrive\Desktop\syncthing-folder\Git-Repos\medicareChatbot\data\pdfs"

os.makedirs(dst_root, exist_ok=True)

count = 0
for root, _, files in os.walk(src_root):
    for f in files:
        if f.lower().endswith(".pdf"):
            src = os.path.join(root, f)
            # prefix the folder name to avoid name collisions
            subdir = os.path.basename(root)
            new_name = f"{subdir}_{f}"
            dst = os.path.join(dst_root, new_name)

            shutil.copy2(src, dst)
            count += 1
            print(f"Copied: {new_name}")

print(f"\n✅ {count} PDF files copied to {dst_root}")
