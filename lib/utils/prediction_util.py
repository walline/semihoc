import torch

def fuse_predictions(softmax,
                     hierarchy,
                     multi_classes,
                     children_maps,
                     group_sizes,
                     path_indices,
                     flat2node_map,
                     uncertainty_method,
                     uncertainty_args,
                     enable_root=False,
                     ):

    max_height = len(softmax)
    n_samples = softmax[0].size(0)

    device = softmax[0].device

    comp_sums = []

    for depth in range(max_height - 1):

        height = max_height - depth - 1

        n_parents = len(multi_classes[depth])

        single_element_mask = group_sizes[depth] == 1
        children_map = children_maps[depth]

        p = softmax[height-1]

        result, p_comp = uncertainty_method(p,
                                            children_map,
                                            group_sizes[depth],
                                            n_samples,
                                            n_parents,
                                            device=device,
                                            **uncertainty_args)

        mapped_single_mask = single_element_mask[children_map]

        p_comp[:, single_element_mask] = 0.0
        result[:, mapped_single_mask] = 1.0

        comp_sums.append(p_comp)
        softmax[height-1].copy_(result)

    expanded_probs = [p[:, path_indices[i]] for i, p in enumerate(reversed(softmax))]
    stacked_probs = torch.stack(expanded_probs, dim=-1)
    cumulative_probs = torch.cumprod(stacked_probs, dim=-1)
    
    results = []

    for height in range(max_height):

        n_classes = len(multi_classes[-height-1])
        depth = max_height - height - 1

        intermediate_prod = torch.zeros(n_samples, n_classes, device=device)
        intermediate_prod[:, path_indices[depth]] = cumulative_probs[:, :, depth]

        if height > 0:
            intermediate_prod = intermediate_prod * comp_sums[depth]

        results.append(intermediate_prod)

    results = torch.cat(results, dim=1)

    results_merged = torch.zeros(n_samples, len(hierarchy.id_node_list), device=device)
    
    results_merged.scatter_add_(1, flat2node_map.expand(n_samples, -1), results)

    # TODO: maybe this can be implemented more cleanly, we should also allow for other root models
    if enable_root:
        root_idx = hierarchy.id_node_list.index("root")
        eps = 1e-9
        p = softmax[-1]
        entropy = -(p * (p + eps).log()).sum(dim=1)
        p_root = entropy / (1.0 + entropy)
        scale = (1.0 - p_root).unsqueeze(1)
        results_merged = results_merged * scale
        results_merged[:, root_idx] = p_root

    psums = torch.sum(results_merged, dim=-1)

    assert torch.allclose(psums, torch.ones_like(psums), atol=1e-4)

    return results_merged
