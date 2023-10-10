from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch
import torchmetrics
from tqdm.auto import tqdm
import wandb
import yaml

from senseiv2 import load_model
from data.dataset import SEnSeIv2Dataset
from data import transforms as trf
from utils import ambiguous_mse_loss, ambiguous_crossentropy_loss, SoftSpatialSegmentationLoss, MappedClassMetric, InformationMetric, Entropy



def construct_data(config):
    """
    Constructs the training and validation datasets
    """

    if not config['MULTIMODAL']: # Only allow multispectral (no SAR/DEM etc.)
        allowed_band_types = ['TOA Reflectance','TOA Normalised Brightness Temperature']
    else:
        allowed_band_types = None

    if not config.get('SEnSeIv2',False):
        train_band_selecter = 'all'
        valid_band_selecter = 'all'
    else:
        train_band_selecter = (config['MIN_BANDS'],config['MAX_BANDS'])
        valid_band_selecter = 'all'

    train_augmentations = [
                trf.Base(config['PATCH_SIZE']),
                trf.Sometimes(0.1,trf.Quantize(0.01)),
                trf.Sometimes(0.02,trf.Quantize(0.04)),
                trf.Sometimes(0.05,trf.Bandwise_salt_and_pepper(0.01,0.01,pepp_value=0.01,salt_value=1.5)),
                trf.Sometimes(0.1,trf.Bandwise_salt_and_pepper(0.01,0.01,pepp_value=0.05,salt_value=1.3)),
                trf.Sometimes(0.1,trf.Bandwise_salt_and_pepper(0.01,0.01,pepp_value=0.1,salt_value=1.1)),
                trf.Sometimes(0.1,trf.Salt_and_pepper(0.01,0.01,pepp_value=0.05,salt_value=1.3)),
                trf.Sometimes(0.25,trf.White_noise(sigma=0.03)),
                trf.Sometimes(0.25,trf.Chromatic_scale(factor_min=0.98, factor_max=1.05)),
                trf.Sometimes(0.25,trf.Chromatic_shift(shift_min=-0.02,shift_max=0.02))
    ]



    if config.get('ADD_NODATA',False):
        train_augmentations.extend([
                trf.Sometimes(0.1,trf.No_data_edges(N=30,p=0.25)), # often, but not too many no-data values
                trf.Sometimes(0.005,trf.No_data_edges(N=200,p=0.25)), # rare, but lots of no-data values
        ])

    valid_augmentations = [
        trf.Base(config['PATCH_SIZE'],fixed=True)
        ]
    
    # Section to deal with different class structures needed for different model runs
    if config['CLASSES'] == 4:
        class_map = OrderedDict()
        class_map['clear']=[0,1,2]
        class_map['thin']=[3]
        class_map['thick']=[4]
        class_map['shadow']=[5]
        train_augmentations += [trf.Class_map(class_map)]
        valid_augmentations += [trf.Class_map(class_map)]
        
    elif config['CLASSES'] == 2:
        class_map = OrderedDict()
        class_map['noncloud']=[0,1,2,5,6]
        class_map['cloud']=[3,4]
        train_augmentations += [trf.Class_map(class_map)]
        valid_augmentations += [trf.Class_map(class_map)]
    else: #7 classes, do nothing
        pass

    # mask/metadata files for ambiguous case
    if config['loss']=='ambiguous_crossentropy_loss':
        im_f,mask_f,metadata_f = 'image.npy','mask_ambiguous.npy','metadata_ambiguous.json'
    else:
        im_f,mask_f,metadata_f = 'image.npy','mask.npy','metadata.json'


    train_dataset = SEnSeIv2Dataset(
            config['TRAIN_DIRS'], 
            config['PATCH_SIZE'],
            transform=train_augmentations,
            band_selection=train_band_selecter,
            allowed_band_types=allowed_band_types,
            im_filename=im_f,
            mask_filename=mask_f,
            metadata_filename=metadata_f
    )
    valid_dataset = SEnSeIv2Dataset(
            config['VALID_DIRS'], 
            config['PATCH_SIZE'],
            transform=valid_augmentations,
            band_selection=valid_band_selecter,
            allowed_band_types=allowed_band_types,
            im_filename=im_f,
            mask_filename=mask_f,
            metadata_filename=metadata_f    
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['BATCH_SIZE'],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=train_dataset.collate_fn,
        drop_last=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config['BATCH_SIZE'],
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=valid_dataset.collate_fn,
    )
    return train_loader, valid_loader

def construct_model(config):

    # Load checkpoint if given
    if config.get('CHECKPOINT', False):
        weights = torch.load(config['CHECKPOINT'])
    else:
        weights = None

    # Use helper function to construct model
    model = load_model(config,weights=weights,device=DEVICE)
    print(model)
    
    return model

# WARNING: Metrics specific to cloud masking. This will need to be changed
# if you want to use this code for anything else.
def construct_metrics(config):
    if config['CLASSES'] == 7:
        train_metrics = {
            'T_acc': MappedClassMetric(
                torchmetrics.classification.MulticlassAccuracy(num_classes=4,average='micro'),
                [0,0,0,1,2,3,np.nan],
                argmax_preds=True,
                argmax_labels=True,
            ).to(DEVICE),
            'T_clear_recall': MappedClassMetric(
                torchmetrics.classification.BinaryRecall(),
                [1,1,1,0,0,0,0],
                argmax_preds=True,
                argmax_labels=True,
            ).to(DEVICE),
            'T_thin_recall': MappedClassMetric(
                torchmetrics.classification.BinaryRecall(),
                [0,0,0,1,0,0,0],
                argmax_preds=True,
                argmax_labels=True,
            ).to(DEVICE),
            'T_thick_recall': MappedClassMetric(
                torchmetrics.classification.BinaryRecall(),
                [0,0,0,0,1,0,0],
                argmax_preds=True,
                argmax_labels=True,
            ).to(DEVICE),
            'T_shadow_recall': MappedClassMetric(
                torchmetrics.classification.BinaryRecall(),
                [0,0,0,0,0,1,0],
                argmax_preds=True,
                argmax_labels=True,
            ).to(DEVICE),
            'T_info': InformationMetric().to(DEVICE),
            'T_entropy': Entropy().to(DEVICE)
        }
        if config.get('ADD_NODATA',False):
            train_metrics['T_nodata_recall'] = MappedClassMetric(
                torchmetrics.classification.BinaryRecall(),
                [0,0,0,0,0,0,1],
                argmax_preds=True,
                argmax_labels=True,
            ).to(DEVICE)

    elif config['CLASSES'] == 4: # Assumes class structure of S2CloudSen12: [0,1,2,3] = [clear, thick, thin, shadow]
        train_metrics = {
            'T_acc': MappedClassMetric(
                torchmetrics.classification.MulticlassAccuracy(num_classes=4,average='micro'),
                [0,1,2,3],
                argmax_preds=True,
                argmax_labels=True,
            ).to(DEVICE),
            'T_clear_recall': MappedClassMetric(
                torchmetrics.classification.BinaryRecall(),
                [1,0,0,0],
                argmax_preds=True,
                argmax_labels=True,
            ).to(DEVICE),
            'T_thin_recall': MappedClassMetric(
                torchmetrics.classification.BinaryRecall(),
                [0,0,1,0],
                argmax_preds=True,
                argmax_labels=True,
            ).to(DEVICE),
            'T_thick_recall': MappedClassMetric(
                torchmetrics.classification.BinaryRecall(),
                [0,1,0,0],
                argmax_preds=True,
                argmax_labels=True,
            ).to(DEVICE),
            'T_shadow_recall': MappedClassMetric(
                torchmetrics.classification.BinaryRecall(),
                [0,0,0,1],
                argmax_preds=True,
                argmax_labels=True,
            ).to(DEVICE),
            'T_entropy': Entropy().to(DEVICE)
        }
    elif config['CLASSES'] == 2: # Assumes simple cloud/non-cloud mask [noncloud, cloud]
        train_metrics = {
            'T_acc': MappedClassMetric(
                torchmetrics.classification.MulticlassAccuracy(num_classes=2,average='micro'),
                [0,1],
                argmax_preds=True,
                argmax_labels=True,
            ).to(DEVICE),
            'T_noncloud_recall': MappedClassMetric(
                torchmetrics.classification.BinaryRecall(),
                [1,0],
                argmax_preds=True,
                argmax_labels=True,
            ).to(DEVICE),
            'T_cloud_recall': MappedClassMetric(
                torchmetrics.classification.BinaryRecall(),
                [0,1],
                argmax_preds=True,
                argmax_labels=True,
            ).to(DEVICE),
            'T_entropy': Entropy().to(DEVICE)
    }

    valid_metrics = {}
    for k,v in train_metrics.items(): #change 'T' to 'V' in names
        valid_metrics[k.replace('T_','V_')] = train_metrics[k].clone().to(DEVICE)
    
    if config['CLASSES'] == 7:
        valid_metrics['Confusion'] = MappedClassMetric(
            torchmetrics.classification.MulticlassConfusionMatrix(num_classes=4),
            [0,0,0,1,2,3,np.nan],
            class_names=['clear','thin','thick','shadow'],
            argmax_preds=True,
            argmax_labels=True,
        ).to(DEVICE)
    elif config['CLASSES'] == 4: # Assumes class structure of S2CloudSen12: [0,1,2,3] = [clear, thick, thin, shadow]
        valid_metrics['Confusion'] = MappedClassMetric(
            torchmetrics.classification.MulticlassConfusionMatrix(num_classes=4),
            [0,1,2,3],
            class_names=['clear','thick','thin','shadow'],
            argmax_preds=True,
            argmax_labels=True,
        ).to(DEVICE)
    elif config['CLASSES'] == 2: # Assumes simple cloud/non-cloud mask [clear, cloud]
        valid_metrics['Confusion'] = MappedClassMetric(
            torchmetrics.classification.MulticlassConfusionMatrix(num_classes=2),
            [0,1],
            class_names=['noncloud','cloud'],
            argmax_preds=True,
            argmax_labels=True,
        ).to(DEVICE)
    return train_metrics, valid_metrics

def plot_batch(imgs,preds,labels,step):
    fig,ax = plt.subplots(1,2)
    plot_mask = np.argmax(preds[0,...].cpu().detach().numpy(),axis=0).astype('float')
    plot_true = np.argmax(labels[0,...].cpu().detach().numpy(),axis=0).astype('float')
    ax[0].imshow(plot_true,cmap='jet',vmin=0,vmax=6)
    ax[1].imshow(plot_mask,cmap='jet',vmin=0,vmax=6)
    rgb = imgs[0,[2,1,0],...].permute(1,2,0).cpu().detach().numpy()
    rgb = (rgb-rgb.min())/(rgb.max()-rgb.min())
    ax[0].imshow(np.clip(rgb*1.4,0,1),alpha=0.85)
    ax[1].imshow(np.clip(rgb*1.4,0,1),alpha=0.85)
    
    # Set axes to be invisible
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
        a.set_frame_on(False)
        
    fig.tight_layout()
    plt.savefig('debugging/test{}.png'.format(step))
    plt.close()


if __name__=='__main__':

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load config
    config_file = sys.argv[1]
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config['device'] = DEVICE

    # Ask user if overwriting of model files is ok
    if os.path.exists('./models/{}/'.format(config['NAME'])):
        while True:
            answer = input("""
            Model directory for {} already exists. 
            Continuing will lead to multiple model files being written to the same 
            directory, and the latest model file will be lost. 
            
            Continue? (y/n)
            """.format(config['NAME']))
            if answer=='y':
                break
            elif answer=='n':
                sys.exit()
    else:
        os.makedirs('./models/{}/'.format(config['NAME']), exist_ok=True)

    wandb_project = config.get('PROJECT','senseiv2')

    # Initialize wandb and make model directory
    wandb.init(project=wandb_project, config=config)

    # build datasets
    train,val = construct_data(config)

    # build model
    model = construct_model(config)

    # build metrics
    train_metrics, valid_metrics = construct_metrics(config)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['LR'][0], eps=config['EPSILON'], weight_decay=config['WEIGHT_DECAY'])
    accum_steps = config['ACCUMULATE_STEPS'][0]  # Number of batches to accumulate gradients over

    # Loss
    if config['LOSS']=='crossentropy_loss':
        loss = torch.nn.CrossEntropyLoss()
    if config['LOSS']=='ambiguous_mse_loss':
        loss = ambiguous_mse_loss
    elif config['LOSS']=='ambiguous_crossentropy_loss':
        loss = ambiguous_crossentropy_loss
    elif config['LOSS']=='spatial_smoothing_loss':
        loss = SoftSpatialSegmentationLoss(
            method=config['LOSS_METHOD'],
            classes=[0,1,2,3],
            kernel_radius=config['LOSS_KERNEL_RADIUS'],
            kernel_sigma=config['LOSS_KERNEL_SIGMA'],
            device=DEVICE
        )
        
    recovery_loss = torch.nn.MSELoss()
    if config.get('RECOVERY_MODULE', False):
        with open(config['RECOVERY_MODULE']) as f:
            recovery_module_config = yaml.load(f, Loader=yaml.FullLoader)
        recovery_sampling_rate = recovery_module_config['sampling_rate']

    # train model
    best_val_loss = 1e10

    # max_steps_per_epoch set because when training on many datasets, the total dataset 
    # length becomes way too large for one epoch. This is a hacky way to limit the number
    # of batches per epoch, and is set to roughly 2-3 times the total training dataset 
    # size of CloudSen12. The tqdm progress bar will still show the total number of
    # batches in the dataset, but the actual number of batches per epoch will be
    # limited to max_steps_per_epoch.
    max_steps_per_epoch = int(20000/config['BATCH_SIZE']) 
    if max_steps_per_epoch<len(train):
        print('WARNING: max_steps_per_epoch set to {} to limit epoch size.'.format(max_steps_per_epoch))

    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(config['EPOCHS']):
        for j,phase in enumerate(config['PHASES']):
            if epoch==phase:
                print('NEW TRAINING PHASE: LR - {}, EFFECTIVE BATCH SIZE - {}'.format(config['LR'][j], config['BATCH_SIZE']*config['ACCUMULATE_STEPS'][j]))
                for p_group in optimizer.param_groups:
                    p_group['lr'] = config['LR'][j]
                accum_steps = config['ACCUMULATE_STEPS'][j]

        print(f'Epoch {epoch+1}/{config["EPOCHS"]}')
        train_loss = []
        for i,batch in enumerate(tqdm(train)):

            # cancel training if max steps per epoch is reached
            if i>max_steps_per_epoch:
                break

            global_step = epoch*min(len(train),max_steps_per_epoch)+i
            # move to device
            batch['inputs'][0] = batch['inputs'][0].to(DEVICE)
            batch['labels'] = batch['labels'].to(DEVICE)

            # forward pass
            preds = model(*batch['inputs'])
            optimizer.zero_grad()

            # If using recovery module during training, split preds.
            # Only needed during training, not evaluation
            if model.recovery_module is not None:
                preds, recovered = preds
                # downsample image
                sampled_inputs = batch['inputs'][0][:,:,::recovery_sampling_rate,::recovery_sampling_rate]
                recovery_reg = recovery_loss(recovered,sampled_inputs)
            else:
                recovery_reg = 0
            
            if config['L1_REG'] > 0:
                l1_reg = sum(p.abs().sum() for p in model.senseiv2.parameters())
            else:
                l1_reg = 0

            if config['LOSS']=='spatial_smoothing_loss':
                argmax_labels = torch.argmax(batch['labels'],dim=1,keepdim=True)
                train_loss.append(loss(preds, argmax_labels,None))
            else:
                train_loss.append(loss(preds, batch['labels']))

            for metric in train_metrics.values():
                metric.update(preds, batch['labels'])


            # Keep regularization terms and training loss term separate for better logging     
            if global_step>=config['RECOVERY_WARMUP_STEPS']:
                total_step_loss = train_loss[-1]/accum_steps + config['L1_REG']*l1_reg + config['RECOVERY_LOSS_FACTOR']*recovery_reg
            else:
                total_step_loss = ((global_step/config['RECOVERY_WARMUP_STEPS'])**2)*train_loss[-1]/accum_steps + config['RECOVERY_LOSS_FACTOR']*recovery_reg

            total_step_loss.backward()

            # Logging
            wandb.log(
                {'T_loss':train_loss[-1]} | {k:v.compute() for k,v in train_metrics.items()},
                step=global_step
            )
            if config['RECOVERY_LOSS_FACTOR']>0:
                wandb.log({'Recovery Loss':recovery_reg}, step=global_step)


            if (i+1)%accum_steps==0 or i==len(train)-1 or i==max_steps_per_epoch-1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1) # Gradient clipping
                optimizer.step()
                train_loss = []

        # Logging of training metrics at end of epoch
        wandb.log({k:v.compute() for k,v in train_metrics.items()}, step=global_step)

        # Validation
        val_loss = []
        with torch.no_grad():
            for batch in val:
                batch['inputs'][0] = batch['inputs'][0].to(DEVICE)
                batch['labels'] = batch['labels'].to(DEVICE)
                preds = model(*batch['inputs'])
                if isinstance(preds,tuple): # Get rid of recovered image if using recovery module
                    preds = preds[0]
                val_loss.append(loss(preds, batch['labels']))
                for metric in valid_metrics.values():
                    metric.update(preds, batch['labels'])
            val_loss = torch.mean(torch.stack(val_loss))

        # Logging
        wandb.log(
            {'lr':optimizer.param_groups[0]['lr'], 'batch_size':config['BATCH_SIZE']*accum_steps} | 
            {'val_loss':val_loss} |
            {k:v.compute() for k,v in valid_metrics.items()}, 
            step=global_step
            )

        # Reset metrics
        for metric in train_metrics.values():
            metric.reset()
        for metric in valid_metrics.values():
            metric.reset()

        # Save model
        if val_loss<best_val_loss:
            best_val_loss = val_loss
            val_loss = val_loss.cpu().detach().numpy()
            torch.save(model.state_dict(), './models/{}/{}-{}.pt'.format(config['NAME'],str(epoch),str(np.round(val_loss,4))))

        torch.save(model.state_dict(), './models/{}/latest.pt'.format(config['NAME']))

