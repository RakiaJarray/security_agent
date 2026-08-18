import os
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"  # timeout plus généreux (défaut ~10s)

from huggingface_hub import snapshot_download, login

login(token="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")  # ton token, ou utilise HF_TOKEN en variable d'environnement

local_dir = snapshot_download(
    repo_id="jayzou3773/CloudAnoBench",
    repo_type="dataset",
    local_dir="./cloudanobench_raw"
)
print("Téléchargé dans:", local_dir)