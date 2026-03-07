import torch

def gen_logit_scores(logits, indices):

    inverse_mask = torch.ones(logits.size(-1), dtype=bool)
    inverse_mask[indices] = False

    other_logits = logits[:, inverse_mask]
    sum_other_logits = other_logits.sum(dim=-1)

    scores = {}

    scores["logitsum"] = sum_other_logits

    return scores


def gen_scores(p):

    scores = {}

    psum = torch.sum(p, dim=-1, keepdim=True)
    comp_sum = 1 - psum
    p_norm = p / psum

    n_classes = p.size(-1)
    n_samples = p.size(0)

    if n_classes == 1:
        entropies = torch.ones(n_samples, dtype=torch.float)
    else:
        entropies = -1.0 * torch.sum(p_norm * torch.log(p_norm), dim=-1)

    p_max, _ = torch.max(p, dim=-1)

    scores["psum"] = comp_sum.squeeze()
    scores["entropy"] = entropies
    scores["pmax"] = 1 - p_max
    scores["entplussum"] = entropies + comp_sum.squeeze()
    scores["massentropy"] = entropies/psum.squeeze()

    return scores

def get_score_keys():
    return ["psum", "entropy", "pmax", "entplussum", "massentropy"]

def get_logit_score_keys():
    return ["logitsum"]


def dirichlet_uncertainty(p,
                          children_map,
                          group_sizes,
                          n_samples,
                          n_parents,
                          device="cpu",
                          k=1.0,
                          ):

    eps = 1e-12
    
    group_sums = torch.zeros(n_samples, n_parents, device=device)
    group_sums.scatter_add_(1, children_map.expand(n_samples, -1), p + eps)
    p_norm = (p + eps) / group_sums[:, children_map]

    validate_tensor(p_norm)

    x = torch.zeros(n_samples, n_parents, device=device)

    plog = torch.log(p_norm + 1e-7)

    assert torch.isfinite(plog).all()

    x.scatter_add_(1, children_map.expand(n_samples, -1), plog)

    assert torch.isfinite(x).all()
    
    x = x * k

    c = group_sizes * k * torch.log(group_sizes)

    assert torch.isfinite(c).all()
    
    x = x + c

    assert torch.isfinite(x).all()
    
    x = torch.exp(x)
    x = torch.clamp(x, 0.0, 1.0)
    
    mapped_x = x[:, children_map]
    p_comp = x
    result = p_norm * (1 - mapped_x)

    # TODO: remove assertions when we are confident
    validate_tensor(p_comp)
    validate_tensor(result)

    result_sums = torch.zeros(n_samples, n_parents, device=device)
    result_sums.scatter_add_(1, children_map.expand(n_samples, -1), result)
    final_sums = result_sums + x
    assert torch.allclose(final_sums, torch.ones_like(final_sums))

    return result, p_comp

def dirichlet_normalized(p,
                         children_map,
                         group_sizes,
                         n_samples,
                         n_parents,
                         device="cpu",
                         k=1.0,
                         ):

    eps = 1e-12

    group_sums = torch.zeros(n_samples, n_parents, device=device)
    group_sums.scatter_add_(1, children_map.expand(n_samples, -1), p + eps)
    p_norm = (p + eps) / group_sums[:, children_map]

    validate_tensor(p_norm)

    k_norm = k / group_sizes

    x = torch.zeros(n_samples, n_parents, device=device)

    plog = torch.log(p_norm + 1e-7)

    assert torch.isfinite(plog).all()

    x.scatter_add_(1, children_map.expand(n_samples, -1), plog)

    assert torch.isfinite(x).all()
    
    x = x * k_norm.view(1, -1)

    c = group_sizes * k_norm * torch.log(group_sizes)

    assert torch.isfinite(c).all()
    
    x = x + c

    assert torch.isfinite(x).all()
    
    x = torch.exp(x)
    x = torch.clamp(x, 0.0, 1.0)
    
    mapped_x = x[:, children_map]
    p_comp = x
    result = p_norm * (1 - mapped_x)

    # TODO: remove assertions when we are confident
    validate_tensor(p_comp)
    validate_tensor(result)

    result_sums = torch.zeros(n_samples, n_parents, device=device)
    result_sums.scatter_add_(1, children_map.expand(n_samples, -1), result)
    final_sums = result_sums + x
    assert torch.allclose(final_sums, torch.ones_like(final_sums))

    return result, p_comp

def pmax_uncertainty(p,
                     children_map,
                     group_sizes,
                     n_samples,
                     n_parents,
                     device="cpu",
                     **kwargs,
                     ):

    eps = 1e-12

    group_sums = torch.zeros(n_samples, n_parents, device=device)
    group_sums.scatter_add_(1, children_map.expand(n_samples, -1), p + eps)
    p_norm = (p + eps) / group_sums[:, children_map]

    validate_tensor(p_norm)

    x = torch.zeros(n_samples, n_parents, device=device)

    x.scatter_reduce_(1, children_map.expand(n_samples, -1), p, reduce="amax", include_self=False)

    assert torch.isfinite(x).all()

    x = (1 - x) / (1 - 1 / group_sizes.view(1, -1) + eps)

    assert torch.isfinite(x).all()

    x = torch.clamp(x, 0.0, 1.0)

    mapped_x = x[:, children_map]
    p_comp = x
    result = p_norm * (1 - mapped_x)

    # TODO: remove assertions when we are confident
    validate_tensor(p_comp)
    validate_tensor(result)

    result_sums = torch.zeros(n_samples, n_parents, device=device)
    result_sums.scatter_add_(1, children_map.expand(n_samples, -1), result)
    final_sums = result_sums + x
    assert torch.allclose(final_sums, torch.ones_like(final_sums))

    return result, p_comp

def compprob(p,
             children_map,
             group_sizes,
             n_samples,
             n_parents,
             device="cpu",
             **kwargs,
             ):

    group_sums = torch.zeros(n_samples, n_parents, device=device)
    group_sums.scatter_add_(1, children_map.expand(n_samples, -1), p)

    p_comp = 1.0 - group_sums

    return p, p_comp

def entcompprob(p,
                children_map,
                group_sizes,
                n_samples,
                n_parents,
                device="cpu",
                **kwargs,
                ):

    eps = 1e-12

    validate_tensor(p)

    group_sums = torch.zeros(n_samples, n_parents, device=device)
    group_sums.scatter_add_(1, children_map.expand(n_samples, -1), p + eps)
    p_norm = (p + eps) / (group_sums[:, children_map] + eps)

    validate_tensor(p_norm)

    x = torch.zeros(n_samples, n_parents, device=device)

    x.scatter_add_(1, children_map.expand(n_samples, -1), -1.0 * p_norm * torch.log(p_norm + eps))

    assert torch.isfinite(x).all()

    total_sums = group_sums + x
    result = p / total_sums[:, children_map]

    p_comp = x / total_sums

    # TODO: remove assertions when we are confident
    validate_tensor(p_comp)
    validate_tensor(result)

    result_sums = torch.zeros(n_samples, n_parents, device=device)
    result_sums.scatter_add_(1, children_map.expand(n_samples, -1), result)
    final_sums = result_sums + p_comp
    assert torch.allclose(final_sums, torch.ones_like(final_sums), atol=1e-4)

    return result, p_comp


def validate_tensor(tensor):
    assert not torch.isnan(tensor).any(), "Tensor contains NaN values!"
    assert not torch.isinf(tensor).any(), "Tensor contains Inf values!"
    assert (tensor >= 0.0).all(), "Tensor contains values less than 0.0!"
    assert (tensor <= 1.0).all(), "Tensor contains values greater than 1.0!"
