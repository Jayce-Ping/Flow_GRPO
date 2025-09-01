from huggingface_hub import upload_folder
from datetime import datetime
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
    folder_path = "logs"
    path_in_repo = "" # As root dir
    repo_id = "Jayce-Ping/Flux-GRPO"  # Replace with your Hugging Face repo ID

    if not label:
        label = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    commit_message = f"Update checkpoints - {label}"

    upload_checkpoint_dir(folder_path, path_in_repo, repo_id, commit_message)