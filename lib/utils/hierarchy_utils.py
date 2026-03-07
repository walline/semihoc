import torch
import itertools
from lib.hierarchy_metrics import calc_hdists
from collections import defaultdict
import json


def get_avg_hdist(hdist_mat):
    hdist_total = 0.
    count = 0.
    for i in range(hdist_mat.shape[0]):
        for j in range(hdist_mat.shape[1]):
            hdist_total += (i+j)*hdist_mat[i,j]
            count += hdist_mat[i,j]
    return hdist_total/count


def get_path_indices(hierarchy, multi_classes, height):

    max_depth = hierarchy._max_depth - height

    indices = [[] for _ in range(max_depth)]

    index = len(multi_classes) - height - 1

    for class_ in multi_classes[index]:
        parents = hierarchy.node_ancestors[class_][1:].copy()
        parents_names = [hierarchy.id_node_list[parent] for parent in parents]
        remainder = max_depth - len(parents_names)
        parents_names = parents_names + remainder * [class_]

        for depth, parent in enumerate(parents_names):
            indices[depth].append(multi_classes[depth].index(parent))

    return indices

def get_children_indices(hierarchy, multi_classes, height):

    assert height > 0

    children_indices = []

    for c_ in multi_classes[-height-1]:

        local_children = []
        
        try:
            children_names = hierarchy.parent2children[c_]
        except KeyError:
            children_names = [c_]

        for name in children_names:
            idx = multi_classes[-height].index(name)
            local_children.append(idx)

        children_indices.append(local_children)

    return children_indices


def get_hdist_matrix(hierarchy, flat_classes, return_pair=False):

    class_list = hierarchy.id_node_list
    class_idxs = range(len(class_list))

    pairs = itertools.product(class_idxs, repeat=2)
    a, b = zip(*pairs)
    a, b = torch.tensor(a, dtype=torch.long), torch.tensor(b, dtype=torch.long)

    gt_dists, pred_dists = calc_hdists(a, b, hierarchy)

    class_to_idx = _value_to_indices(flat_classes)

    mat_size = len(flat_classes)

    hdist_mat = torch.zeros(mat_size, mat_size, dtype=torch.long)

    pred_dists_mat = torch.zeros(mat_size, mat_size, dtype=torch.long)
    gt_dists_mat = torch.zeros(mat_size, mat_size, dtype=torch.long)

    for i in range(len(a)):
        class_a = a[i]
        class_b = b[i]
        pred_dist = pred_dists[i]
        gt_dist = gt_dists[i]

        idx_a = class_to_idx[int(class_a)]
        idx_b = class_to_idx[int(class_b)]

        for ii in idx_a:
            for jj in idx_b:
                gt_dists_mat[ii, jj] = gt_dist
                pred_dists_mat[ii, jj] = pred_dist

    if return_pair:
        return gt_dists_mat.float(), pred_dists_mat.float()
    else:
        hdist_mat = gt_dists_mat + pred_dists_mat
        return hdist_mat.float()


def get_expected_hdist(softmax_preds, hdist_mat):

    n_classes = softmax_preds.size(-1)

    expected_hdists = torch.zeros_like(softmax_preds)
    
    for c in range(n_classes):
        hdists = hdist_mat[:, c]
        pred_hdists = softmax_preds * hdists
        e_hd = torch.sum(pred_hdists, dim=-1)
        expected_hdists[:, c] = e_hd

    # the LLM proposes the below solution but it does not seem to be correct
    # expected_hdists = torch.matmul(softmax_preds, hdist_mat)
    
    return expected_hdists
    
def _value_to_indices(lst):
    value_to_indices = defaultdict(list)
    
    # Iterate over the list with index
    for idx, value in enumerate(lst):
        value_to_indices[value].append(idx)
    
    return value_to_indices

def get_multidepth_classes(hierarchy, train_classes):

    max_depth = hierarchy._max_depth
    
    c = [set() for _ in range(hierarchy._max_depth)]
        
    for train_class in train_classes:
        parents = hierarchy.node_ancestors[train_class][1:].copy()
        parents_names = [hierarchy.id_node_list[parent] for parent in parents]
        remainder = max_depth - len(parents)
        parents_names = parents_names + remainder * [train_class]

        for depth, parent in enumerate(parents_names):
            c[depth].add(parent)

    multi_classes = [sorted(list(x)) for x in c]
            
    return multi_classes


def get_target_distributions(hierarchy, multi_classes, device="cpu"):

    multi_classes_reversed = list(reversed(multi_classes))
    
    n_classes = len(hierarchy.id_node_list)

    target_distributions = []
    for classes in multi_classes_reversed:
        init_tensor = torch.zeros((n_classes, len(classes)),
                                  dtype=torch.float,
                                  device=device)
        target_distributions.append(init_tensor)

    for i, c in enumerate(hierarchy.id_node_list):

        ancestors_idxs = hierarchy.node_ancestors[c]
        ancestors_names = [hierarchy.id_node_list[ii] for ii in ancestors_idxs]
        path_names = ancestors_names + [c]

        descendants = hierarchy.get_descendants(c)

        path_names = path_names + descendants

        assert len(path_names) == len(set(path_names))

        for j, _ in enumerate(target_distributions):
            classes = multi_classes_reversed[j]
            matching_classes = [x for x in path_names if x in classes]
            matching_idxs = [classes.index(x) for x in matching_classes]
            n_matching = len(matching_idxs)

            target_distributions[j][i, matching_idxs] = 1.0 / n_matching

    for dist in target_distributions:
        target_sum = torch.sum(dist, dim=-1)
        assert torch.allclose(target_sum, torch.ones_like(target_sum), atol=1e-3)

    return target_distributions


def get_leaf_mask(hierarchy, device="cpu"):

    mask = torch.zeros(len(hierarchy.id_node_list), dtype=torch.bool, device=device)

    for i, c in enumerate(hierarchy.id_node_list):

        if c in hierarchy.train_classes:
            mask[i] = True

    return mask


def get_children_and_group_maps(hierarchy, multi_classes, device="cpu"):

    children_maps = []
    group_sizes = []

    max_height = hierarchy._max_depth

    for depth in range(max_height - 1):

        height = max_height - depth - 1
        n_children = len(multi_classes[depth+1])
        n_parents = len(multi_classes[depth])
        children_map = torch.full((n_children,), -1, dtype=torch.long, device=device)
        sizes = torch.zeros(n_parents, device=device)
        children_indices = get_children_indices(hierarchy, multi_classes, height)

        for group_idx, indices in enumerate(children_indices):
            children_map[torch.tensor(indices)] = group_idx
            sizes[group_idx] = len(indices)

        children_maps.append(children_map)
        group_sizes.append(sizes)

    # note that children_maps and group sizes are indexed by depth
    # instead of height, because there is no entry for height=0
    return children_maps, group_sizes


def get_leaves_from_json(hierarchy_fn):

    def get_leaves(node):
        if not node.get("children"):
            return [node["name"]]
        leaves = []
        for child in node["children"]:
            leaves.extend(get_leaves(child))
        return leaves

    with open(hierarchy_fn, "r") as file:
        tree = json.load(file)

    leaf_nodes = get_leaves(tree)

    return sorted(leaf_nodes)
