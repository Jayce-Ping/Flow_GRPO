from huggingface_hub import snapshot_download

api_key="hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

snapshot_download(
    repo_id='RevisualR1/umm',
    local_dir='flow_grpo/dataset/umm',
    token=api_key
)
