from datasets.basic_dataset_scaffold import BaseDataset
import os
import numpy as np

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
    image_sourcepath  = datapath
    #Find available data classes.
    image_classes = sorted([x for x in os.listdir(image_sourcepath)])
    genera = np.unique([x.split('_')[0] for x in image_classes])
    #Make a index-to-labelname conversion dict.
    conversion    = {i:image_classes[i] for i in range(len(image_classes))}
    back_conversion    = {v:k for k,v in conversion.items()}

    #Generate a list of tuples (class_label, image_path)
    image_list    = {back_conversion[key]:sorted([image_sourcepath+'/'+key+'/'+x for x in os.listdir(image_sourcepath+'/'+key) ]) for key in image_classes}
    image_list    = [[(key,img_path) for img_path in image_list[key]] for key in image_list.keys()]
    image_list    = [x for y in image_list for x in y]

    #Image-dict of shape {class_idx:[list of paths to images belong to this class] ...}
    image_dict    = {}
    for key, img_path in image_list:
        if not key in image_dict.keys():
            image_dict[key] = []
        image_dict[key].append(img_path)

    keys = sorted(list(image_dict.keys()))

    #Take all genera in staphylininae subfamily, and put them in train set, 
    # 
    train_genera = ['Acylophorus','Bisnius','Cafius','Creophilus','Dinothenarus','Emus',
    'Erichsonius','Euryporus','Gabrius','Gabronthus','Heterothops','Neobisnius','Ocypus',
    'Ontholestes','Philonthus','Platydracus','Quedius','Remus','Staphylinus','Tasgius','Velleius']
    test_genera = [x for x in genera if x not in train_genera]
    
    #Following "Deep Metric Learning via Lifted Structured Feature Embedding", we use the first half of classes for training.
    train = [back_conversion[x] for x in image_classes if x.split('_')[0] in train_genera]
    test = [back_conversion[x] for x in image_classes if x.split('_')[0] in test_genera]
    train_image_dict, test_image_dict = {key:image_dict[key] for key in train},{key:image_dict[key] for key in test}
    val_image_dict = None
    train_conversion = {i:conversion[key] for i,key in enumerate(train)}
    test_conversion  = {i:conversion[key] for i,key in enumerate(test)}

    train_dataset = BaseDataset(train_image_dict, opt)
    test_dataset   = BaseDataset(test_image_dict,   opt, is_validation=True)
    eval_dataset  = BaseDataset(train_image_dict, opt, is_validation=True)
    eval_train_dataset  = BaseDataset(train_image_dict, opt, is_validation=False)

    print('\nDataset Setup:\nUsing Train-Val Split: {0}\n#Classes: Train ({1}) | Val ({2}) | Test ({3})\n'.format(opt.use_tv_split, len(train_image_dict), len(val_image_dict) if val_image_dict else 'X', len(test_image_dict)))

    train_dataset.conversion = train_conversion
    test_dataset.conversion   = test_conversion
    eval_dataset.conversion  = test_conversion
    eval_train_dataset.conversion  = train_conversion

    return {'training':train_dataset, 'validation':None, 'testing':test_dataset, 'evaluation':eval_dataset, 'evaluation_train':eval_train_dataset}