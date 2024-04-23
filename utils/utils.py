import torch


def get_k_lowest_nonzero_indices(tensor, k):
    # Flatten the tensor
    flat_tensor = tensor.flatten()

    # Get the indices of non-zero elements
    non_zero_indices = torch.nonzero(flat_tensor, as_tuple=True)[0]

    # Select the non-zero elements
    non_zero_elements = torch.index_select(flat_tensor, 0, non_zero_indices)

    # Get the indices of the k lowest elements
    k_lowest_values, k_lowest_indices = torch.topk(non_zero_elements, k, largest=False)

    # Get the original indices
    k_lowest_original_indices = non_zero_indices[k_lowest_indices]

    return k_lowest_original_indices
