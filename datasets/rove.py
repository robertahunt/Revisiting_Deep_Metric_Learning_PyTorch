from lib2to3.pytree import Base
from datasets.basic_dataset_scaffold import BaseDataset
import os
import numpy as np

import pandas as pd

from ete3 import Tree
from tqdm import tqdm
from glob import glob
from pathlib import Path



# modified from ete3 codebase
def my_convert_to_ultrametric(tree, tree_length=None, strategy="fixed_child"):
    """
    .. versionadded: 2.1

    Converts a tree into ultrametric topology (all leaves must have
    the same distance to root). Note that, for visual inspection
    of ultrametric trees, node.img_style["size"] should be set to
    0.
    """

    # Could something like this replace the old algorithm?
    # most_distant_leaf, tree_length = self.get_farthest_leaf()
    # for leaf in self:
    #    d = leaf.get_distance(self)
    #    leaf.dist += (tree_length - d)
    # return

    # get origin distance to root
    dist2root = {tree: 0.0}
    for node in tree.iter_descendants("levelorder"):
        dist2root[node] = dist2root[node.up] + node.dist

    # get tree length by the maximum
    if not tree_length:
        tree_length = max(dist2root.values())
    else:
        tree_length = float(tree_length)

    # converts such that starting from the leaves, each group which can have
    # the smallest step, does. This ensures all from the same genus are assumed the same
    # space apart
    if strategy == "fixed_child":
        step = 1.0

        # pre-calculate how many splits remain under each node
        node2max_depth = {}
        for node in tree.traverse("postorder"):
            if not node.is_leaf():
                max_depth = max([node2max_depth[c] for c in node.children]) + 1
                node2max_depth[node] = max_depth
            else:
                node2max_depth[node] = 1
        node2dist = {tree: -1.0}
        # modify the dist property of nodes
        for node in tree.iter_descendants("levelorder"):
            node.dist = tree_length - node2dist[node.up] - node2max_depth[node] * step

            # print(node,node.dist, node.up)
            node2dist[node] = node.dist + node2dist[node.up]

    return tree



def get_T_matrix(classes, conversion, phylogeny):
    t = phylogeny.copy()
    t = my_convert_to_ultrametric(t)
    no_classes = len(classes)
    T = np.zeros((no_classes, no_classes))

    t.prune(classes)
    assert len(t.get_leaves()) == len(classes)
    # First, assign mu to each of the leaves/species
    classes = []

    debug = False
    if debug:
        # don't bother calculating T since it takes so damn long for large phylogenies
        return t.get_leaf_names(), np.diagflat(np.ones(no_classes))

    leaves = t.get_leaves()
    leaves = sorted(leaves, key=lambda x: x.name)
    for i in tqdm(range(len(leaves))):
        leaf1 = leaves[i]
        classes += [leaf1.name]
        j = 0
        for j in range(i, len(leaves)):
            leaf2 = leaves[j]
            ancestor = leaf1.get_common_ancestor(leaf2)
            T[i, j] = ancestor.get_distance(t)
            T[j, i] = T[i, j]

    class_indices = [conversion[x] for x in classes]
    T = pd.DataFrame(T, index=class_indices, columns=class_indices)
    return T


def Give(opt, datapath):
    """
    This function generates a training, testing and evaluation dataloader for Metric Learning on the Rove dataset.

    Args:
        opt: argparse.Namespace, contains all traininig-specific parameters.
    Returns:
        dict of PyTorch datasets for training, testing and evaluation.
    """
    assert (
        opt.use_tv_split == True
    ), "This dataset uses a train, validation and test set"

    image_sourcepath = Path(datapath)
    # Find available data classes.
    image_classes = sorted(
        np.unique([Path(x).name for x in glob(str(image_sourcepath / "*" / "*"))])
    )
    train_classes = sorted(
        [Path(x).name for x in glob(str(image_sourcepath / "train" / "*"))]
    )
    val_classes = sorted(
        [Path(x).name for x in glob(str(image_sourcepath / "val" / "*"))]
    )
    test_classes = sorted(
        [Path(x).name for x in glob(str(image_sourcepath / "test" / "*"))]
    )

    # Make a index-to-labelname conversion dict.
    conversion = {i: image_classes[i] for i in range(len(image_classes))}
    back_conversion = {v: k for k, v in conversion.items()}

    # Generate a list of tuples (class_label, image_path)
    image_list = {
        back_conversion[key]: sorted(glob(str(image_sourcepath / "*" / key / "*.jpg")))
        for key in image_classes
    }
    image_list = [
        [(key, img_path) for img_path in image_list[key]] for key in image_list.keys()
    ]
    image_list = [x for y in image_list for x in y]

    # Image-dict of shape {class_idx:[list of paths to images belong to this class] ...}
    image_dict = {}
    for key, img_path in image_list:
        if not key in image_dict.keys():
            image_dict[key] = []
        image_dict[key].append(img_path)



    train = [back_conversion[x] for x in train_classes]
    val = [back_conversion[x] for x in val_classes]
    test = [back_conversion[x] for x in test_classes]
    assert len(set(image_classes)) == len(set(train + val + test))

    train_image_dict = {key: [x for x in image_dict[key] if x.split('/')[-3] == 'train'] for key in train}
    val_image_dict = {key: [x for x in image_dict[key] if x.split('/')[-3] == 'val'] for key in val}
    test_image_dict = {key: [x for x in image_dict[key] if  x.split('/')[-3] == 'test' ] for key in test}

    train_dataset = BaseDataset(train_image_dict, opt)
    val_dataset = BaseDataset(val_image_dict, opt, is_validation=True)
    test_dataset = BaseDataset(test_image_dict, opt, is_validation=True)
    eval_dataset = BaseDataset(train_image_dict, opt, is_validation=True)
    eval_train_dataset = BaseDataset(train_image_dict, opt, is_validation=False)

    if 'Genus' in datapath:
        phylogeny = Tree(str(image_sourcepath / 'phylogenyGenus.nh'))
    else:
        phylogeny = Tree(str(image_sourcepath / 'phylogeny.nh'))
    train_dataset.T = get_T_matrix(train_classes, back_conversion, phylogeny)
    val_dataset.T = get_T_matrix(val_classes, back_conversion, phylogeny)
    test_dataset.T = get_T_matrix(test_classes, back_conversion, phylogeny)
    eval_dataset.T = get_T_matrix(train_classes, back_conversion, phylogeny)
    eval_train_dataset.T = get_T_matrix(train_classes, back_conversion, phylogeny)

    print(
        "\nDataset Setup:\nUsing Train-Val Split: {0}\n#Classes: Train ({1}) | Val ({2}) | Test ({3})\n".format(
            opt.use_tv_split,
            len(train_image_dict),
            len(val_image_dict) if val_image_dict else "X",
            len(test_image_dict),
        )
    )

    train_dataset.conversion = conversion
    val_dataset.conversion = conversion
    test_dataset.conversion = conversion
    eval_dataset.conversion = conversion
    eval_train_dataset.conversion = conversion

    return {
        "training": train_dataset,
        "validation": val_dataset,
        "testing": test_dataset,
        "evaluation": eval_dataset,
        "evaluation_train": eval_train_dataset,
    }
