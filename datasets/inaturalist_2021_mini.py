from datasets.basic_dataset_scaffold import BaseDataset
import os
import numpy as np

from glob import glob


def Give(opt, datapath):
    """
    This function generates a training, testing and evaluation dataloader for Metric Learning on the Rove dataset.
    For Metric Learning, the dataset classes are sorted by name, and the first half used for training while the last half is used for testing.
    So no random shuffling of classes.

    Args:
        opt: argparse.Namespace, contains all traininig-specific parameters.
    Returns:
        dict of PyTorch datasets for training, testing and evaluation.
    """
    assert opt.use_tv_split == False
    image_sourcepath = datapath
    # Find available data classes.
    train_path = os.path.join(image_sourcepath, "train_mini")
    val_path = os.path.join(image_sourcepath, "val")
    image_classes = sorted([x for x in os.listdir(train_path)])

    # Make a index-to-labelname conversion dict.
    conversion = {i: image_classes[i] for i in range(len(image_classes))}
    back_conversion = {v: k for k, v in conversion.items()}

    # Image-dict of shape {class_idx:[list of paths to images belong to this class] ...}
    train_image_dict = {}
    for _class in image_classes:
        train_image_dict[back_conversion[_class]] = glob(
            os.path.join(train_path, _class, "*.jpg")
        )
    test_image_dict = {}
    for _class in image_classes:
        test_image_dict[back_conversion[_class]] = glob(
            os.path.join(val_path, _class, "*.jpg")
        )
    val_image_dict = None

    train_dataset = BaseDataset(train_image_dict, opt)
    test_dataset = BaseDataset(test_image_dict, opt, is_validation=True)
    eval_dataset = BaseDataset(train_image_dict, opt, is_validation=True)
    eval_train_dataset = BaseDataset(train_image_dict, opt, is_validation=False)

    print(
        "\nDataset Setup:\nUsing Train-Val Split: {0}\n#Classes: Train ({1}) | Val ({2}) | Test ({3})\n".format(
            opt.use_tv_split,
            len(train_image_dict),
            len(val_image_dict) if val_image_dict else "X",
            len(test_image_dict),
        )
    )

    train_dataset.conversion = conversion
    test_dataset.conversion = conversion
    eval_dataset.conversion = conversion
    eval_train_dataset.conversion = conversion

    return {
        "training": train_dataset,
        "validation": None,
        "testing": test_dataset,
        "evaluation": eval_dataset,
        "evaluation_train": eval_train_dataset,
    }
