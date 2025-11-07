import os
from datetime import datetime
from huggingface_hub import upload_folder, delete_folder, upload_large_folder
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
        repo_type='model',
        ignore_patterns=["*.tmp", "*.DS_Store", "*.log", "*.ipynb", '**/eval_images/**', "*.png", "*.jpg"],
    )
    print(f"Folder '{folder_path}' successfully uploaded to '{repo_id}'.")

def delete_checkpoint_dir(path_in_repo, repo_id, commit_message="Delete folder", token=None):
    """
    Delete a folder from the Hugging Face Hub.

    Args:
        path_in_repo (str): Path to the folder in the repository to delete.
        repo_id (str): Repository ID on Hugging Face Hub (e.g., "username/repo_name").
        commit_message (str): Commit message for the deletion.
        token (str, optional): Hugging Face authentication token. If None, it uses the locally cached token.
    """
    delete_folder(
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        commit_message=commit_message,
        token=token,
        repo_type='model'
    )
    print(f"Folder '{path_in_repo}' successfully deleted from '{repo_id}'.")

if __name__ == "__main__":
    label = "Upload final basic setting"
    # SAVE_DIR = '/home/users/astar/ares/cp3jia/scratch/Flow_NFT/logs'
    # SAVE_DIR = '/home/users/astar/ares/cp3jia/scratch/Flow_Kontext/logs'
    # SAVE_DIR = '/home/users/astar/cfar/stuchengyou/jcy/Flux_Kontext/logs'
    SAVE_DIR = '/home/users/astar/cfar/stuchengyou/jcy/Paco-GRPO/logs'
    # folder_name = 'consistency_for_editing/Flux-Kontext-GEdit-Split-EN-20251107-000031'
    folder_name = "consistencyReward-subclip/10s-1cot_ppo_10sde_train1_groupstd_train-small-20251106-232135"
    folder_path = os.path.join(SAVE_DIR, folder_name)
    path_in_repo = folder_name
    repo_id = "Jayce-Ping/Flux-NFT"

    if not label:
        label = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    commit_message = f"Update checkpoints{folder_name} - {label}"

    upload_checkpoint_dir(folder_path, path_in_repo, repo_id, commit_message)
    # delete_checkpoint_dir(path_in_repo, repo_id, commit_message)