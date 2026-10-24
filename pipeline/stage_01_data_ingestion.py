"""Stage 1: download the zip from Google Drive and unpack it so that
artifacts/data_ingestion/dataset/ contains the class folders (yes/ and no/)."""
import os
import re
import shutil
import zipfile

from pipeline.common import read_params



def find_dataset_root(base: str, classes: list) -> str:
    """Folder that directly contains every class folder (case-insensitive).
    Works whether the zip has yes/ no/ at the top or nested inside a wrapper folder."""
    wanted = {c.lower() for c in classes}
    for root, dirs, _ in os.walk(base):
        dirs[:] = [d for d in dirs if not d.startswith(("__MACOSX", "."))]   # skip zip junk
        if wanted <= {d.lower() for d in dirs}:
            return root
    raise FileNotFoundError(f"No folder containing sub-folders {sorted(wanted)} found under {base}")



def drive_file_id(url: str):
    m = re.search(r"/d/([\w-]+)", url) or re.search(r"[?&]id=([\w-]+)", url)
    return m.group(1) if m else None


def main():
    p = read_params()
    base = p["data_dir"]
    final = os.path.join(base, "dataset")

    if os.path.isdir(final):
        try:
            find_dataset_root(final, p["classes"])
            print("Dataset already present, skipping download")
            return
        except FileNotFoundError:
            shutil.rmtree(final)

    url = p["data_url"]
    if not url:
        raise SystemExit("Set data_url in params.yaml (zip containing yes/ and no/ folders)")

    import gdown

    tmp = os.path.join(base, "_tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    zip_path = os.path.join(tmp, "data.zip")

    file_id = drive_file_id(url)
    out = gdown.download(id=file_id, output=zip_path, quiet=False) if file_id \
        else gdown.download(url, zip_path, quiet=False)
    if not out or not zipfile.is_zipfile(zip_path):
        raise SystemExit("Download failed or the file is not a zip. Check that the Drive link is shared "
                         "as 'Anyone with the link' and (if Drive's quota message appears) retry later.")

    with zipfile.ZipFile(zip_path) as z:
        z.extractall(tmp)
    os.remove(zip_path)

    root = find_dataset_root(tmp, p["classes"])
    os.makedirs(final)
    for d in os.listdir(root):                       # normalise folder names to lower case: Yes -> yes
        if d.lower() in p["classes"]:
            shutil.move(os.path.join(root, d), os.path.join(final, d.lower()))
    shutil.rmtree(tmp, ignore_errors=True)
    print("Ingestion done:", final)


if __name__ == "__main__":
    main()
