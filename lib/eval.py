from lib import hierarchy_metrics as hm

from lib.utils.hierarchy_utils import get_avg_hdist

def get_results(preds,
                node_labels,
                id_hierarchy,
                dists_mats=None,
                ):

    hmet = hm.HierarchicalPredAccuracy(id_hierarchy, track_hdist=True)

    hmet.update_state(preds.long(),
                      node_labels,
                      dists_mats=dists_mats)

    hd = hmet.result_hierarchy_distances()
    balanced_acc = hmet.result_balanced_accuracy()
    balanced_hdist = hmet.result_balanced_hierarchy_distance()
    class_hdists = hmet.result_class_hdists()

    return {"acc": hmet.result(),
            "balanced_acc": balanced_acc,
            "hdist": hd,
            "avg_hdist": get_avg_hdist(hd),
            "balanced_hdist": balanced_hdist,
            "class_hdists": class_hdists,
            }
