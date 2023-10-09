from pathlib import Path
from tqdm import tqdm
import requests
import tempfile
import sys


def download_file(url, fp, skip_if_exists=True):
    if skip_if_exists and Path(fp).is_file() and Path(fp).stat().st_size > 0:
        return
    r = requests.get(url, stream=True)
    assert r.status_code == 200
    progress_bar = tqdm(
        total=int(r.headers.get("content-length", 0)),
        unit="B",
        unit_scale=True,
        desc=url,
    )
    (path := Path(fp).parent).mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path, delete=False) as f:
        for chunk in r.iter_content(chunk_size=16384):
            progress_bar.update(f.write(chunk))
        f.close()
        Path(f.name).rename(fp)


# print to stderr
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
