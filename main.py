"""==================================================================================================="""
################### LIBRARIES ###################
### Basic Libraries
import warnings

warnings.filterwarnings("ignore")

import cv2
import os, sys, numpy as np, argparse, imp, datetime, pandas as pd, copy
import time, pickle as pkl, random, json, collections
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt

from time import sleep
from tqdm import tqdm
#from carbontracker.tracker import CarbonTracker

import parameters as par
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image



"""==================================================================================================="""
################### INPUT ARGUMENTS ###################
parser = argparse.ArgumentParser()

parser = par.basic_training_parameters(parser)
parser = par.batch_creation_parameters(parser)
parser = par.batchmining_specific_parameters(parser)
parser = par.loss_specific_parameters(parser)
parser = par.wandb_parameters(parser)

##### Read in parameters
opt = parser.parse_args()


"""==================================================================================================="""
### The following setting is useful when logging to wandb and running multiple seeds per setup:
### By setting the savename to <group_plus_seed>, the savename will instead comprise the group and the seed!
if opt.savename == "group_plus_seed":
    if opt.log_online:
        opt.savename = opt.group + "_s{}".format(opt.seed)
    else:
        opt.savename = ""

### If wandb-logging is turned on, initialize the wandb-run here:
if opt.log_online:
    import wandb

    _ = os.system("wandb login {}".format(opt.wandb_key))
    os.environ["WANDB_API_KEY"] = opt.wandb_key
    wandb.init(
        project=opt.project, group=opt.group, name=opt.savename, dir=opt.save_path
    )
    wandb.config.update(opt)


"""==================================================================================================="""
### Load Remaining Libraries that neeed to be loaded after comet_ml
import torch, torch.nn as nn
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")
import architectures as archs
import datasampler as dsamplers
import datasets as datasets
import criteria as criteria
import metrics as metrics
import batchminer as bmine
import evaluation as eval
from utilities import misc
from utilities import logger


"""==================================================================================================="""
full_training_start_time = time.time()


"""==================================================================================================="""
opt.source_path += "/" + opt.dataset
opt.save_path += "/" + opt.dataset

# Assert that the construction of the batch makes sense, i.e. the division into class-subclusters.
assert (
    not opt.bs % opt.samples_per_class
), "Batchsize needs to fit number of samples per class for distance sampling and margin/triplet loss!"

opt.pretrained = not opt.not_pretrained


"""==================================================================================================="""
################### GPU SETTINGS ###########################
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# if not opt.use_data_parallel:
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu[0])


"""==================================================================================================="""
#################### SEEDS FOR REPROD. #####################
torch.backends.cudnn.deterministic = True
np.random.seed(opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)


"""==================================================================================================="""
##################### NETWORK SETUP ##################
opt.device = torch.device("cuda")
model = archs.select(opt.arch, opt)

if opt.fc_lr < 0:
    to_optim = [{"params": model.parameters(), "lr": opt.lr, "weight_decay": opt.decay}]
else:
    all_but_fc_params = [
        x[-1]
        for x in list(
            filter(lambda x: "last_linear" not in x[0], model.named_parameters())
        )
    ]
    fc_params = model.model.last_linear.parameters()
    to_optim = [
        {"params": all_but_fc_params, "lr": opt.lr, "weight_decay": opt.decay},
        {"params": fc_params, "lr": opt.fc_lr, "weight_decay": opt.decay},
    ]

_ = model.to(opt.device)


"""============================================================================"""
#################### DATALOADER SETUPS ##################
dataloaders = {}
datasets = datasets.select(opt.dataset, opt, opt.source_path)

dataloaders["evaluation"] = torch.utils.data.DataLoader(
    datasets["evaluation"], num_workers=opt.kernels, batch_size=opt.bs, shuffle=False
)
dataloaders["testing"] = torch.utils.data.DataLoader(
    datasets["testing"], num_workers=opt.kernels, batch_size=opt.bs, shuffle=False
)
if opt.use_tv_split:
    dataloaders["validation"] = torch.utils.data.DataLoader(
        datasets["validation"],
        num_workers=opt.kernels,
        batch_size=opt.bs,
        shuffle=False,
    )

train_data_sampler = dsamplers.select(
    opt.data_sampler,
    opt,
    datasets["training"].image_dict,
    datasets["training"].image_list,
)
if train_data_sampler.requires_storage:
    train_data_sampler.create_storage(dataloaders["evaluation"], model, opt.device)

dataloaders["training"] = torch.utils.data.DataLoader(
    datasets["training"], num_workers=opt.kernels, batch_sampler=train_data_sampler
)

opt.n_classes = len(dataloaders["training"].dataset.avail_classes)
if "rove" in opt.dataset:
    opt.n_classes = (
        len(dataloaders["training"].dataset.avail_classes)
        + len(dataloaders["testing"].dataset.avail_classes)
        + len(dataloaders["validation"].dataset.avail_classes)
    )


"""============================================================================"""
#################### CREATE LOGGING FILES ###############
sub_loggers = ["Train", "Test", "Model Grad"]
if opt.use_tv_split:
    sub_loggers.append("Val")
LOG = logger.LOGGER(
    opt, sub_loggers=sub_loggers, start_new=True, log_online=opt.log_online
)


"""============================================================================"""
#################### LOSS SETUP ####################
batchminer = bmine.select(opt.batch_mining, opt)
criterion, to_optim = criteria.select(opt.loss, opt, to_optim, batchminer)
_ = criterion.to(opt.device)

if "criterion" in train_data_sampler.name:
    train_data_sampler.internal_criterion = criterion


"""============================================================================"""
#################### OPTIM SETUP ####################
if opt.optim == "adam":
    optimizer = torch.optim.Adam(to_optim)
elif opt.optim == "sgd":
    optimizer = torch.optim.SGD(to_optim, momentum=0.9)
else:
    raise Exception("Optimizer <{}> not available!".format(opt.optim))
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=opt.tau, gamma=opt.gamma
)


"""============================================================================"""
#################### METRIC COMPUTER ####################
opt.rho_spectrum_embed_dim = opt.embed_dim
metric_computer = metrics.MetricComputer(opt.evaluation_metrics, opt)


"""============================================================================"""
################### Summary #########################3
data_text = "Dataset:\t {}".format(opt.dataset.upper())
setup_text = "Objective:\t {}".format(opt.loss.upper())
miner_text = "Batchminer:\t {}".format(
    opt.batch_mining if criterion.REQUIRES_BATCHMINER else "N/A"
)
arch_text = "Backbone:\t {} (#weights: {})".format(
    opt.arch.upper(), misc.gimme_params(model)
)
summary = data_text + "\n" + setup_text + "\n" + miner_text + "\n" + arch_text
print(summary)



"""============================================================================"""
################### Load Model #########################
if len(opt.checkpoint):
    assert os.path.exists(opt.checkpoint), f"Could not find path at {opt.checkpoint}"
    model.load_state_dict(torch.load(opt.checkpoint)['state_dict'])


"""============================================================================"""
################### Produce saliency maps #########################
if opt.gradcam:
    target_layers =[ model.model.layer4[-1].conv3]#   last convolutional layer
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # for each genus - get an example per genus
    image_dict = datasets['validation'].image_dict

    assert len(opt.checkpoint)
    gradcam_folder = os.path.join('/'.join(opt.checkpoint.split('/')[:-1]), 'gradcamconv3_bw')
    if not os.path.exists(gradcam_folder):
        os.makedirs(gradcam_folder)
    
    genera_ids = image_dict.keys()
    for genus_id in genera_ids:
        genus = datasets['validation'].conversion[genus_id]
        image_id = image_dict[genus_id][0][-1]
        inp = datasets['validation'].__getitem__(image_id)[1].unsqueeze(0)

        image = inp[0].permute(1,2,0).detach().cpu().numpy()
        image = (image - image.min()) / (image.max()-image.min())
        output = model(inp.cuda())[0].squeeze().cpu().detach().numpy()
        for latent_var in range(128):
            targets = [ClassifierOutputTarget(latent_var)]
            input_tensor = inp.repeat(len(targets), 1, 1, 1)
            

            saliency = cam(input_tensor=input_tensor,targets=targets)[0]
            visualization = show_cam_on_image(image, saliency, use_rgb = True)
            image_mask = (cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) < 0.75).astype('float')
            bw_vis = image_mask*0.25 + saliency*0.75
            latent_folder = os.path.join(gradcam_folder,str(latent_var))
            sleep(0.01)
            if not os.path.exists(latent_folder):
                os.makedirs(latent_folder)
            
            
            fn = str(np.round((output[latent_var] + 1)*1000, 0)).replace('.','_') + '_' + genus + '.png'
            plot_path = os.path.join(latent_folder, fn)
            plt.figure()
            plt.axis('off')
            plt.imshow(bw_vis, cmap='gray')
            #plt.imshow(visualization, cmap='gray')
            plt.savefig(plot_path, bbox_inches='tight')

    quit()


def plot_inputs(dataloaders):
    
    data_iterator = tqdm(
        dataloaders["training"], desc="Plotting Training...".format(epoch)
    )

    for i, out in enumerate(data_iterator):
        class_labels, input, input_indices = out

        if not os.path.exists(os.path.join(opt.save_path,'train_examples')):
            os.makedirs(os.path.join(opt.save_path,'train_examples'))

        for j in range(input.shape[0]):
            img = input[j].permute(1,2,0).detach().cpu().numpy()
            img = (img - img.min())/(img.max() - img.min()) * 255
            img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(opt.save_path,'train_examples',f'{j}.png'),img.astype('uint8'))
        break
    data_iterator = tqdm(
        dataloaders["validation"], desc="Plotting Validation...".format(epoch)
    )

    for i, out in enumerate(data_iterator):
        class_labels, input, input_indices = out

        if not os.path.exists(os.path.join(opt.save_path,'val_examples')):
            os.makedirs(os.path.join(opt.save_path,'val_examples'))

        for j in range(input.shape[0]):
            img = input[j].permute(1,2,0).detach().cpu().numpy()
            img = (img - img.min())/(img.max() - img.min()) * 255
            img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(opt.save_path,'val_examples',f'{j}.png'),img)
        break

"""============================================================================"""
################### SCRIPT MAIN ##########################
print("\n-----\n")

iter_count = 0
loss_args = {"batch": None, "labels": None, "batch_features": None, "f_embed": None}
#carbon_tracker = CarbonTracker(opt.n_epochs,
#            log_dir='carbontracker/',monitor_epochs=-1)

for epoch in range(opt.n_epochs):
    #carbon_tracker.epoch_start()
    epoch_start_time = time.time()

    if epoch > 0 and opt.data_idx_full_prec and train_data_sampler.requires_storage:
        train_data_sampler.full_storage_update(
            dataloaders["evaluation"], model, opt.device
        )

    opt.epoch = epoch
    ### Scheduling Changes specifically for cosine scheduling
    if opt.scheduler != "none":
        print(
            "Running with learning rates {}...".format(
                " | ".join("{}".format(x) for x in scheduler.get_lr())
            )
        )

    """======================================="""
    if train_data_sampler.requires_storage:
        train_data_sampler.precompute_indices()

    """======================================="""
    ### Train one epoch
    start = time.time()
    _ = model.train()

    loss_collect = []

    #plot_inputs(dataloaders)

    data_iterator = tqdm(
        dataloaders["training"], desc="Epoch {} Training...".format(epoch)
    )

    for i, out in enumerate(data_iterator):
        class_labels, input, input_indices = out

        ### Compute Embedding
        input = input.to(opt.device)
        model_args = {"x": input.to(opt.device)}
        # Needed for MixManifold settings.
        if "mix" in opt.arch:
            model_args["labels"] = class_labels
        embeds = model(**model_args)
        if isinstance(embeds, tuple):
            embeds, (avg_features, features) = embeds

        ### Compute Loss
        loss_args["batch"] = embeds
        loss_args["labels"] = class_labels
        loss_args["f_embed"] = model.model.last_linear
        loss_args["batch_features"] = features
        loss_args["T"] = dataloaders["training"].dataset.T
        loss = criterion(**loss_args)

        if not torch.isnan(loss):
            loss.backward()

            ###
            loss_collect.append(loss.item())

        ###
        iter_count += 1

        if i == len(dataloaders["training"]) - 1:
            data_iterator.set_description(
                "Epoch (Train) {0}: Mean Loss [{1:.4f}]".format(
                    epoch, np.mean(loss_collect)
                )
            )

        """======================================="""
        if train_data_sampler.requires_storage and train_data_sampler.update_storage:
            train_data_sampler.replace_storage_entries(
                embeds.detach().cpu(), input_indices
            )

        ###
        if ((i + 1) % opt.gradient_accumulation_steps == 0) or (
            i == (len(data_iterator) - 1)
        ):
            ### Update network weights!
            optimizer.step()
            optimizer.zero_grad()

            grads = [
                    p.grad.detach().cpu().numpy().flatten()
                    for p in model.parameters()
                    if p.grad is not None
                ]
            if len(grads):
                ### Compute Model Gradients and log them!
                grads = np.concatenate(grads)
                grad_l2, grad_max = np.mean(np.sqrt(np.mean(np.square(grads)))), np.mean(
                    np.max(np.abs(grads))
                )
                LOG.progress_saver["Model Grad"].log("Grad L2", grad_l2, group="L2")
                LOG.progress_saver["Model Grad"].log("Grad Max", grad_max, group="Max")

        if opt.debug:
            break

    result_metrics = {"loss": np.mean(loss_collect)}

    ####
    LOG.progress_saver["Train"].log("epochs", epoch)
    for metricname, metricval in result_metrics.items():
        LOG.progress_saver["Train"].log(metricname, metricval)
    LOG.progress_saver["Train"].log("time", np.round(time.time() - start, 4))

    """======================================="""
    if ((epoch % 10) == 0) or (epoch == opt.n_epochs - 1):
        ### Evaluate Metric for Training & Test (& Validation)
        _ = model.eval()
        print("\nComputing Testing Metrics...")
        eval.evaluate(
            opt.dataset,
            LOG,
            metric_computer,
            [dataloaders["testing"]],
            model,
            opt,
            opt.evaltypes,
            opt.device,
            log_key="Test",
        )
        if opt.use_tv_split:
            print("\nComputing Validation Metrics...")
            eval.evaluate(
                opt.dataset,
                LOG,
                metric_computer,
                [dataloaders["validation"]],
                model,
                opt,
                opt.evaltypes,
                opt.device,
                log_key="Val",
            )
        print("\nComputing Training Metrics...")
        eval.evaluate(
            opt.dataset,
            LOG,
            metric_computer,
            [dataloaders["evaluation"]],
            model,
            opt,
            opt.evaltypes,
            opt.device,
            log_key="Train",
        )

        LOG.update(all=True)

        """======================================="""
        ### Learning Rate Scheduling Step
        if opt.scheduler != "none":
            scheduler.step()

        print("Total Epoch Runtime: {0:4.2f}s".format(time.time() - epoch_start_time))
        print("\n-----\n")

        if opt.debug:
            break
        
        #carbon_tracker.epoch_end()


"""======================================================="""
### CREATE A SUMMARY TEXT FILE
summary_text = ""
full_training_time = time.time() - full_training_start_time
summary_text += "Training Time: {} min.\n".format(np.round(full_training_time / 60, 2))

summary_text += "---------------\n"
for sub_logger in LOG.sub_loggers:
    try:
        metrics = LOG.graph_writer[sub_logger].ov_title
        summary_text += "{} metrics: {}\n".format(sub_logger.upper(), metrics)
    except:
        print(f'Error in logging {sub_logger}')


with open(opt.save_path + "/training_summary.txt", "w") as summary_file:
    summary_file.write(summary_text)
