import os
from datetime import datetime
from huggingface_hub import upload_folder
def upload_checkpoint_dir(folder_path, path_in_repo, repo_id, commit_message="Upload folder", token=None):
    """
    Upload a folder to the Hugging Face Hub.

    Args:
        folder_path (str): Path to the folder to upload.
        repo_id (str): Repository ID on Hugging Face Hub (e.g., "username/repo_name").
        commit_message (str): Commit message for the upload.
        token (str, optional): Hugging Face authentication token. If None, it uses the locally cached token.
    """
    upload_folder(
        folder_path=folder_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        commit_message=commit_message,
        token=token,
        repo_type='model'
    )
    print(f"Folder '{folder_path}' successfully uploaded to '{repo_id}'.")

if __name__ == "__main__":
    label = ""
    folder_name = 'grid-consistency-subclip'
    folder_path = os.path.join("logs", folder_name)
    path_in_repo = folder_name
    repo_id = "Jayce-Ping/Flux-GRPO"

    if not label:
        label = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    commit_message = f"Update checkpoints{folder_name} - {label}"

    upload_checkpoint_dir(folder_path, path_in_repo, repo_id, commit_message)