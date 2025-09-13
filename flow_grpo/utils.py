import re
import base64
from io import BytesIO
from typing import List, Union, Optional, Dict
from itertools import permutations, combinations, chain
import torch.distributed as dist

from PIL import Image
import torch
import numpy as np
import openai
from accelerate import Accelerator

# -------------------------------------Image Utils-------------------------------------

def pil_image_to_base64(image, format="JPEG"):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
    base64_qwen = f"data:image/{format.lower()};base64,{encoded_image_text}"
    return base64_qwen

def tensor_to_pil_image(tensor: torch.Tensor) -> List[Image.Image]:
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)
    
    images = (tensor * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
    images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
    images = [Image.fromarray(image) for image in images]
    images = images
    return images

def numpy_to_pil_image(array: np.ndarray) -> List[Image.Image]:
    if len(array.shape) == 3:
        array = array[np.newaxis, ...]
    
    # Clip and convert to uint8
    if array.max() <= 1.0:
        array = (array * 255).round()
    array = np.clip(array, 0, 255).astype(np.uint8)

    # Convert from NCHW to NHWC if needed
    if array.shape[1] == 3:  # NCHW format
        array = array.transpose(0, 2, 3, 1)  # NCHW -> NHWC

    images = [Image.fromarray(image) for image in array]
    images = images
    return images


def tensor_list_to_pil_image(tensor_list: List[torch.Tensor]) -> List[Image.Image]:
    if not tensor_list:
        return []

    # If all image tensors have the same shape, stack them directly
    if all(tensor.shape == tensor_list[0].shape for tensor in tensor_list):
        batch = torch.stack([
            t if t.dim() == 3 else t.squeeze(0)
            for t in tensor_list
        ], dim=0)
        # Normalize, to uint8
        batch = (batch * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        # NCHW -> NHWC
        if batch.shape[1] == 3:
            batch = batch.transpose(0, 2, 3, 1)
        return [Image.fromarray(img) for img in batch]
    else:
        # Process each tensor individually
        images = []
        for t in tensor_list:
            if t.dim() == 4 and t.shape[0] == 1:
                t = t.squeeze(0)
            img = (t * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            if img.shape[0] == 3:
                img = img.transpose(1, 2, 0)  # CHW -> HWC
            images.append(Image.fromarray(img))
        return images

def numpy_list_to_pil_image(numpy_list: List[np.ndarray]) -> List[Image.Image]:
    if not numpy_list:
        return []
    # If all image arrays have the same shape, stack them directly
    if all(arr.shape == numpy_list[0].shape for arr in numpy_list):
        batch = np.stack([
            arr if arr.ndim == 3 else arr.squeeze(0)
            for arr in numpy_list
        ], axis=0)
        # Normalize, to uint8
        if batch.max() <= 1.0:
            batch = (batch * 255).round()
        batch = np.clip(batch, 0, 255).astype(np.uint8)
        # NCHW -> NHWC
        if batch.shape[1] == 3:
            batch = batch.transpose(0, 2, 3, 1)
        return [Image.fromarray(img) for img in batch]
    else:
        # Process each array individually
        images = []
        for arr in numpy_list:
            if arr.ndim == 4 and arr.shape[0] == 1:
                arr = arr.squeeze(0)
            if arr.max() <= 1.0:
                arr = (arr * 255).round()
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            if arr.shape[0] == 3:
                arr = arr.transpose(1, 2, 0)  # CHW -> HWC
            images.append(Image.fromarray(arr))
        return images


# -------------------------------------Grid Utils-------------------------------------
def divide_prompt(prompt):
    # seqis like ". [TOP-LEFT]:"
    match_sep = re.compile(r"\.\s+[A-Z0-9-\[\]]+:")
    seps = match_sep.findall(prompt)
    # Add '.' for each sentence
    sub_prompts = [
        p + '.' if p.strip()[-1] != '.' else p
        for p in re.split('|'.join(map(re.escape, seps)), prompt)
    ]
    return sub_prompts

def divide_image(image, grid_info : tuple[int, int]):
    assert len(grid_info) == 2, "grid_info must be a tuple of two integers (a, b)"

    a, b = grid_info
    width, height = image.size

    grid_cells = []
    cell_height = height // a
    cell_width = width // b

    # 2x2 grid
    # | 1 | 2 |
    # | 3 | 4 |
    # [
    # (0, 0, cell_width, cell_height),
    # (cell_width, 0, 2 * cell_width, cell_height),
    # (0, cell_height, cell_width, 2 * cell_height),
    # (cell_width, cell_height, 2 * cell_width, 2 * cell_height)
    # ]

    for i in range(a):
        for j in range(b):
            upper = i * cell_height
            left = j * cell_width
            right = left + cell_width
            lower = upper + cell_height
            grid_cells.append(image.crop((left, upper, right, lower)))

    return grid_cells

def extract_grid_info(prompt) -> tuple[int, int]:
    # Grid can be represented as int x int, or int ⨉ int. ⨉ has unicode \u2a09
    match = re.findall(r'(\d+)\s*[x⨉]\s*(\d+)', prompt)
    if len(match) == 0:
        return (1, 1)

    return (int(match[0][0]), int(match[0][1]))


# -------------------------------------OpenAI Utils------------------------------------
def get_yes_cond_prob_from_completion(completion : openai.ChatCompletion, canonicalize=False) -> float:
    if completion is None:
        return 0.0

    logprobs = completion.choices[0].logprobs
    if logprobs:
        # Use logprobs to compute, score = P('yes') / (P('yes') + P('no'))
        # score = 1 / (1 + exp(logprob('no') -  logprob('yes')))
        # Same formular for logits as well. Since the sum term will cancel out.
        # Use uppercase only here.
        if not canonicalize:
            token_logprobs = {t.token: t.logprob for t in logprobs.content[0].top_logprobs}
            yes_logprob = token_logprobs.get('Yes', float('-inf'))
            no_logprob = token_logprobs.get('No', float('-inf'))
            if yes_logprob == float('-inf') and no_logprob == float('-inf'):
                # When inf - inf encountered, give 0.0 score.
                yes_cond_prob = 0.0 # 0.0
            else:
                diff = torch.tensor(yes_logprob - no_logprob, dtype=torch.float64)
                yes_cond_prob = torch.sigmoid(diff).item()
        else:
            # Sum all possible cases together
            # 'yes', 'Yes', 'YES', 'yes ',....
            # 'no', 'No', 'NO',....
            token_probs = {t.token: np.exp(t.logprob, dtype=np.float64) for t in logprobs.content[0].top_logprobs}
            
            # Vectorized computation
            tokens = np.array(list(token_probs.keys()))
            probs = np.array(list(token_probs.values()))
            
            # Strip and lower the tokens for matching
            tokens_stripped = np.array([token.strip().lower() for token in tokens])
            
            yes_mask = tokens_stripped == "yes"
            no_mask = tokens_stripped == "no"
            
            yes_prob_sum = probs[yes_mask].sum()
            no_prob_sum = probs[no_mask].sum()
            
            total = yes_prob_sum + no_prob_sum

            if total == 0.0:
                yes_cond_prob = 0.0
            else:
                yes_cond_prob = yes_prob_sum / total
    else:
        # log_prob cannot be derived here. Return 0.0.
        # TODO
        yes_cond_prob = 0.0

    return yes_cond_prob





# -------------------------------------Reward Computation Utils---------------------------------------
def is_symmetric_matrix(matrix: np.ndarray):
    """
        Check if the matrix is symmetric
        Args:
            matrix: square numpy array
        Returns:
            bool: True if symmetric, False otherwise
    """
    matrix = np.array(matrix)
    if matrix.shape[0] != matrix.shape[1]:
        # Must be square
        return False

    return np.all(matrix == matrix.T)

def is_antisymmetric_matrix(matrix: np.ndarray, diagonal_zero=True):
    """
        Check if the matrix is anti-symmetric
        Args:
            matrix: square numpy array
            diagonal_zero: if True, check if diagonal elements are zero, else ignore diagonal
        Returns:
            bool: True if anti-symmetric, False otherwise
    """
    matrix = np.array(matrix)
    n = matrix.shape[0]
    if matrix.shape[0] != matrix.shape[1]:
        # Must be square
        return False

    summation = matrix.T + matrix
    if diagonal_zero:
        # Check if all elements are zero
        return np.all(summation == 0)
    else:
        # Assign diagonal to zero and check
        summation[np.diag_indices_from(summation)] = 0
        if np.any(summation != 0):
            return False

    return True

def is_transitive_matrix(matrix: np.ndarray, return_violations=False):
    """
        Check if the matrix is transitive
        Args:
            matrix: square numpy array with binary values (0 or 1)
        Returns:
            bool: True if transitive, False otherwise
    """
    matrix = np.array(matrix)
    n = len(matrix)
    if matrix.shape[0] != matrix.shape[1]:
        # Must be square
        return False
    
    if not np.all(np.isin(matrix, [0, 1])):
        # Must be binary
        raise ValueError("`transitiveMatrixQ` requires matrix must be binary (0 or 1)")

    # Check transitivity: if A[i][j] == 1 and A[j][k] == 1, then A[i][k] must be 1
    violations = []
    for i,j,k in permutations(range(n), 3):
        # Check all 3-tuples
        if matrix[i][j] == 1 and matrix[j][k] == 1 and matrix[i][k] != 1:
            if not return_violations:
                return False

            violations.append((i,j,k))


    if return_violations:
        return len(violations) == 0, violations

    return len(violations) == 0


# -----------------------------------Tensor Gathering Utils---------------------------------------
def all_gather_tensor_list(
        accelerator: Accelerator,
        tensor_list: List[torch.Tensor],
        dtype: Optional[torch.dtype]=None,
        device: Union[str, torch.device]=torch.device("cpu")
    ) -> List[torch.Tensor]:
    """
    Gather a list of tensors from all processes, each process has a list of tensors.
    Each tensor can have a different shape (e.g., (C, H, W)).

    Args:
        accelerator (`Accelerator`): Accelerator object
        tensor_list (`List[torch.Tensor]`): list of tensors to gather, each tensor can have different shape but same dimension,  for example, [(3, 64, 64), (3, 128, 128), ...]. Each list can have different length on different processes.
        dtype (`torch.dtype`, *optional*): dtype of the gathered tensors, if None, use the dtype of the first tensor in tensor_list
        device (`Union[str, torch.device]`, *optional*, defaults to `torch.device("cpu")`): device of the gathered tensors

    Returns:
        gathered_tensors (`List[torch.Tensor]`): tensors from all processes, concatenated in rank order
    """
    if not tensor_list:
        return []
    
    assert all(isinstance(t, torch.Tensor) for t in tensor_list), "All elements in tensor_list must be torch.Tensor"
    assert all(t.dim() == tensor_list[0].dim() for t in tensor_list), "All tensors must have the same number of dimensions"

    tensor_dim = tensor_list[0].dim()
    tensor_dtype = tensor_list[0].dtype if dtype is None else dtype
    device = torch.device(device)

    # Step 1: Gather lengths of tensor_list from all ranks
    local_length = torch.tensor([len(tensor_list)], device=accelerator.device, dtype=torch.long)
    gathered_lengths = [torch.zeros(1, dtype=torch.long, device=accelerator.device) for _ in range(accelerator.num_processes)]
    dist.all_gather(gathered_lengths, local_length)
    gathered_lengths = [int(length.item()) for length in gathered_lengths]

    # Step 2: Gather shapes of each tensor in tensor_list from all ranks
    local_shapes = torch.tensor([list(t.shape) for t in tensor_list], device=accelerator.device, dtype=torch.long)
    gathered_shapes = [
        torch.zeros((length, tensor_dim), dtype=torch.long, device=accelerator.device)
        for length in gathered_lengths
    ]
    dist.all_gather(gathered_shapes, local_shapes)
    gathered_shapes = [shapes.cpu() for shapes in gathered_shapes]  # Move to CPU to save some GPU memory

    # Compute the total length of flattened tensors for each rank, [rank0_total_length, rank1_total_length, ...]
    flat_lengths = [
        sum(int(shape.prod().item()) for shape in this_rank_shapes)
        for this_rank_shapes in gathered_shapes
    ]

    # Step 3: Gather all tensors by flattening and concatenating
    local_flat_tensor = torch.cat([t.flatten() for t in tensor_list], dim=0)
    gathered_flat_tensors = [
        torch.zeros(length, dtype=tensor_dtype, device=accelerator.device)
        for length in flat_lengths
    ]
    dist.all_gather(gathered_flat_tensors, local_flat_tensor)
    gathered_flat_tensors = [t.cpu() for t in gathered_flat_tensors]  # Move to CPU to save some GPU memory

    # Step 4: Reconstruct the original tensors from gathered shapes and flattened tensors
    gathered_tensors = []
    for rank, (this_rank_shapes, this_rank_flat_tensor) in enumerate(zip(gathered_shapes, gathered_flat_tensors)):
        offset = 0
        for shape in this_rank_shapes:
            length = int(shape.prod().item())
            # Reshape and move to the specified device
            this_tensor = this_rank_flat_tensor[offset:offset+length].reshape(shape.tolist()).to(device)
            gathered_tensors.append(this_tensor)
            offset += length

    return gathered_tensors