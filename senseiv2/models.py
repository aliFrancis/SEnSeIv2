import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from transformers import logging, SegformerForSemanticSegmentation
import yaml

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    import segmentation_models_pytorch as smp
# Get rid of annoying warnings from transformers
logging.set_verbosity_error()

class FullModel(torch.nn.Module):
    """
    Used to combine SEnSeIv2 with segmentation model
    """
    def __init__(self,senseiv2,segmenter,recovery_module=None, logits_called=False):
        super(FullModel, self).__init__()
        self.senseiv2 = senseiv2
        self.segmenter = segmenter
        self.recovery_module = recovery_module
        self.logits_called = logits_called

        if self.senseiv2 is None and self.recovery_module is not None:
            raise ValueError('If senseiv2 is None, recovery_module must also be None')
        self.it = 0

    def forward(self, bands, descriptors):     
        band_types = [d['band_type'] for d in descriptors[0]]

        if self.senseiv2 is not None:
            embeddings,descriptors = self.senseiv2(bands,descriptors)
            if self.training and self.recovery_module is not None:
                recovered = self.recovery_module(embeddings,descriptors)
        else:
            embeddings = bands
        
        if self.logits_called:
            logits = self.segmenter(embeddings).logits
        else:
            logits = self.segmenter(embeddings)

        resampled = torch.nn.functional.interpolate(logits, embeddings.shape[-2:], mode='bilinear', align_corners=False)
        outputs = torch.nn.functional.softmax(resampled, dim=1)
        if self.training and self.recovery_module is not None:
            return outputs, recovered
        else:
            return outputs

class SEnSeIv2(torch.nn.Module):
    """
    SEnSeIv2 model. Constructed as a sequence of blocks, defined in a config file.

    SEnSeIv2 should begin with a descriptor embedding block. This defines how the 
    information about each band (stored in a dictionary) is held in a vector format,
    to be ingested into the model.
    """
    def __init__(self,config):
        super(SEnSeIv2, self).__init__()
        self.descriptor_style = config['descriptors']['style']
        self.device = config['device']
        if self.descriptor_style == 'SEnSeIv1':
            self.descriptor_vectorizer = SEnSeIv1Descriptors(config['descriptors'],device=self.device)
        elif self.descriptor_style == 'SEnSeIv2':
            self.descriptor_vectorizer = SEnSeIv2Descriptors(config['descriptors'],device=self.device)
        elif self.descriptor_style == 'embedding':
            self.descriptor_vectorizer = EmbeddedDescriptors(config['descriptors'],device=self.device)
        else:
            raise ValueError('Unknown descriptor style: {}'.format(self.descriptor_style))
        self.descriptor_size = self.descriptor_vectorizer.get_output_size()
        self.blocks = torch.nn.ModuleList(self._construct_blocks(config['blocks']))
        self.output_size = self.blocks[-1].get_output_size()

    def forward(self, bands, descriptors):
        descriptors = self.descriptor_vectorizer(descriptors)
        for i,block in enumerate(self.blocks):
            bands,descriptors = block(bands,descriptors)
        return bands, descriptors

    def _construct_blocks(self, config):
        blocks = []
        next_input_descriptor_dims = self.descriptor_size
        for i, block_config in enumerate(config):
            block = self._construct_block(block_config,input_descriptor_dims=next_input_descriptor_dims,device=self.device)
            next_input_descriptor_dims = block.get_output_size()
            blocks.append(block)
        return blocks

    def _construct_block(self, block_config, input_descriptor_dims=None, device='cuda'):
        block_type = block_config['type']
        if block_type == 'GLOBAL_STATS':
            return GlobalStats(block_config,input_descriptor_dims=input_descriptor_dims,device=device)
        elif block_type == 'FCL':
            return FCLBlock(block_config,input_descriptor_dims=input_descriptor_dims,device=device)
        elif block_type == 'PERMUTED_FCL':
            return PermutedFCLBlock(block_config,input_descriptor_dims=input_descriptor_dims,device=device)
        elif block_type == 'ATTENTION':
            return AttentionBlock(block_config,input_descriptor_dims=input_descriptor_dims,device=device)
        elif block_type == 'BAND_EMBEDDING':
            return BandEmbedding(block_config,input_descriptor_dims=input_descriptor_dims,device=device)
        elif block_type == 'BAND_MULTIPLICATION':
            return BandMultiplication(block_config,input_descriptor_dims=input_descriptor_dims,device=device)
        else:
            raise ValueError(f'Unknown block type {block_type}')

class SEnSeIv2Descriptors(torch.nn.Module):
    """
    Converts a list of descriptor dictionaries into a tensor.

    This is necessarily application-specific. The current implementation is
    focussed on cloud masking in multispectral instruments, with some additional
    functionality for cocontemporaneous SAR and DEM data.

    The design principle here is that the dictionaries are relatively easy for a 
    user of the cloud mask algorithm to engineer, requiring only a few lines of
    configuration. The downside is that the implementation is not very flexible,
    and will require editing for other applications.
    """

    def __init__(self,config,input_descriptor_dims=None,device='cuda'):
        super(SEnSeIv2Descriptors,self).__init__()
        self.config = config
        self.N_embeddings = config['N_embeddings']
        self.input_descriptor_dims = input_descriptor_dims
        self.device = device

    def forward(self, descriptor_dicts):
        output_descriptors = torch.zeros(len(descriptor_dicts),len(descriptor_dicts[0]),self.config['final_size']).to(self.device)
        for b,batch_dicts in enumerate(descriptor_dicts):
            for i,d_dict in enumerate(batch_dicts):
                band_type = d_dict['band_type']
                if band_type=='TOA Reflectance' or band_type=='TOA Normalised Brightness Temperature':
                    min_wavelength = d_dict['min_wavelength']
                    max_wavelength = d_dict['max_wavelength']

                    min_wavelength_enc = self.position_encoding(min_wavelength,self.N_embeddings)
                    max_wavelength_enc = self.position_encoding(max_wavelength,self.N_embeddings)

                    output_descriptors[b,i,:self.N_embeddings] = min_wavelength_enc
                    output_descriptors[b,i,self.N_embeddings:self.N_embeddings*2] = max_wavelength_enc

                    output_descriptors[b,i,self.N_embeddings*2] = 1
                    if band_type=='TOA Normalised Brightness Temperature':
                        output_descriptors[b,i,self.N_embeddings*2+3] = 1
                    if d_dict.get('multitemporal',False):
                        output_descriptors[b,i,-1] = 1

                elif 'SAR' in band_type:
                    output_descriptors[b,i,self.N_embeddings*2+1] = 1
                    if 'VV' in band_type:
                        output_descriptors[b,i,:self.N_embeddings] = 1
                    elif 'VH' in band_type:
                        output_descriptors[b,i,self.N_embeddings:self.N_embeddings*2] = 1
                    elif 'Angle' in band_type:
                        output_descriptors[b,i,:self.N_embeddings*2] = 1
                elif 'DEM' in band_type:
                    output_descriptors[b,i,self.N_embeddings*2+2] = 1
                elif band_type=='fill':
                    output_descriptors[b,i,:] = 0
                else:
                    raise ValueError('Unknown band type: {}'.format(band_type))
        return output_descriptors

    def position_encoding(self, val, N):
        # Altered from https://github.com/jadore801120/attention-is-all-you-need-pytorch
        '''Init the sinusoid position encoding table '''

        position_enc = np.array(
            [val / np.power(10000, 2*i/N) for i in range(N)]
        )
        position_enc[0::2] = np.sin(position_enc[0::2])
        position_enc[1::2] = np.cos(position_enc[1::2])
        return torch.from_numpy(position_enc).type(torch.FloatTensor).to(self.device)

    def get_output_size(self):
        return self.config['final_size']



class SEnSeIv1Descriptors(torch.nn.Module):
    """
    Descriptor style used in SEnSeIv1. This is a simple log transform of the input.

    Does not support non-multispectral (e.g. SAR/DEM) inputs.
    """

    def __init__(self,config,input_descriptor_dims=None,device='cuda'):
        super(SEnSeIv1Descriptors,self).__init__()
        self.config = config
        self.input_descriptor_dims = input_descriptor_dims
        self.device = device
        
    def forward(self, descriptor_dicts):
        output_descriptors = torch.zeros(len(descriptor_dicts),len(descriptor_dicts[0]),3).to(self.device)
        for b,batch_dicts in enumerate(descriptor_dicts):
            for i,d_dict in enumerate(batch_dicts):
                band_type = d_dict['band_type']
                if band_type=='TOA Reflectance' or band_type=='TOA Normalized Brightness Temperature':
                    min_wavelength = d_dict['min_wavelength']
                    max_wavelength = d_dict['max_wavelength']
                    central_wavelength = min_wavelength + (max_wavelength-min_wavelength)/2
                    output_descriptors[b,i,0] = np.log(min_wavelength-300)-2
                    output_descriptors[b,i,1] = np.log(central_wavelength-300)-2
                    output_descriptors[b,i,2] = np.log(max_wavelength-300)-2
                elif band_type=='fill':
                    output_descriptors[b,i,:] = 0
                else:
                    raise ValueError('Unknown band type {}'.format(band_type))
        return output_descriptors

    def get_output_size(self):
        return 3

class EmbeddedDescriptors(torch.nn.Module):
    """
    EXPERIMENTAL

    Converts a list of descriptor dictionaries into a tensor.

    This is necessarily application-specific. The current implementation is
    focussed on cloud masking in multispectral instruments, with some additional
    functionality for cocontemporaneous SAR and DEM data.

    The design principle here is that the dictionaries are relatively easy for a 
    user of the cloud mask algorithm to engineer, requiring only a few lines of
    configuration. The downside is that the implementation is not very flexible,
    and will require editing for other applications.
    """

    def __init__(self,config,input_descriptor_dims=None, device='cuda'):
        super(EmbeddedDescriptors,self).__init__()
        self.config = config
        self.N_embeddings = config['N_embeddings']
        self.frequencies = torch.nn.Parameter(0.02*torch.rand(config['N_embeddings']))
        self.phase_offsets = torch.nn.Parameter(0.1*torch.rand(config['N_embeddings'])-0.05)
        self.input_descriptor_dims = input_descriptor_dims
        self.device = device

    def forward(self, descriptor_dicts):
        output_descriptors = torch.zeros(len(descriptor_dicts),len(descriptor_dicts[0]),self.config['final_size']).to(self.device)
        for b,batch_dicts in enumerate(descriptor_dicts):
            for i,d_dict in enumerate(batch_dicts):
                band_type = d_dict['band_type']
                if band_type=='TOA Reflectance' or band_type=='TOA Normalized Brightness Temperature':
                    min_wavelength = d_dict['min_wavelength']
                    max_wavelength = d_dict['max_wavelength']

                    # Calculate the embeddings
                    output_descriptors[b,i,:self.N_embeddings] = \
                        torch.sin(
                            self.frequencies*(torch.tensor([min_wavelength]*self.N_embeddings).to(self.device) \
                            + self.phase_offsets)
                            )
                    output_descriptors[b,i,self.N_embeddings:2*self.N_embeddings] = \
                        torch.sin(
                            self.frequencies*(torch.tensor([max_wavelength]*self.N_embeddings).to(self.device) \
                            + self.phase_offsets)
                            )

                    output_descriptors[b,i,self.N_embeddings*2] = 1
                elif 'SAR' in band_type:
                    output_descriptors[b,i,self.N_embeddings*2+1] = 1
                    if 'VV' in band_type:
                        output_descriptors[b,i,:self.N_embeddings] = 1
                    elif 'VH' in band_type:
                        output_descriptors[b,i,self.N_embeddings:self.N_embeddings*2] = 1
                    elif 'Angle' in band_type:
                        output_descriptors[b,i,:self.N_embeddings*2] = 1
                elif 'DEM' in band_type:
                    output_descriptors[b,i,self.N_embeddings*2+2] = 1
                elif band_type=='fill':
                    output_descriptors[b,i,:] = 0
                else:
                    raise ValueError('Unknown band type')
        return output_descriptors

    def get_output_size(self):
        return self.config['final_size']


class GlobalStats(torch.nn.Module):
    """
    Calculates various percentiles of the input bands, and concatenates them to the
    descriptors. This allows SEnSeI to learn something about the reflectance values in
    the bands, without having to consider the entire image space (would be too memory intensive).
    """
    def __init__(self,config,input_descriptor_dims=None, device='cuda'):
        super(GlobalStats,self).__init__()
        self.device = device
        self.percentiles = torch.tensor(config['percentiles'],device=self.device)
        self.input_descriptor_dims = input_descriptor_dims

    def forward(self, bands, descriptors):
        stats = torch.zeros((bands.shape[0],bands.shape[1],len(self.percentiles)),device=self.device)
        reshaped_bands = bands.reshape(bands.shape[0],bands.shape[1],-1)
        for i,p in enumerate(self.percentiles):
            stats[:,:,i] = torch.quantile(reshaped_bands, p, dim=2)
        return bands, torch.cat((descriptors,stats),dim=2)

    def get_output_size(self):
        return self.input_descriptor_dims + len(self.percentiles)

class FCLBlock(torch.nn.Module):
    """
    Simple fully-connected layer block. Set of FCLs with LayerNorm and ReLU activations.
    Optional skips will form residual connections across each layer (applied immediately after
    the convolution).
    """
    def __init__(self,config,input_descriptor_dims=None, device='cuda'):
        super(FCLBlock,self).__init__()
        self.layer_sizes = config['layer_sizes']
        self.skips = config['skips']
        self.input_descriptor_dims = input_descriptor_dims
        self.device = device

        self.layers = torch.nn.ModuleList(self._get_layers())
        
        # init weights
        for layer in self.layers:
            self._init_weights(layer)

    def _init_weights(self,layer):
        if type(layer) == torch.nn.Linear:
            torch.nn.init.kaiming_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)
        # if type(layer) == torch.nn.LayerNorm:
        #     layer.bias.data.fill_(0.01)
        #     layer.weight.data.fill_(1)

    def _get_layers(self):

        layers = []
        for i,layer_size in enumerate(self.layer_sizes):
            # order of blocks is: Norm -> ReLU -> Linear (-> skip)
            if i == 0:
                layers.append(torch.nn.LayerNorm(self.input_descriptor_dims))
                layers.append(torch.nn.ReLU())
                layers.append(torch.nn.Linear(self.input_descriptor_dims,layer_size))
            if i > 0:
                layers.append(torch.nn.LayerNorm(self.layer_sizes[i-1]))
                layers.append(torch.nn.ReLU())
                layers.append(torch.nn.Linear(self.layer_sizes[i-1],layer_size))
            if self.skips:
                layers.append(torch.nn.Identity()) #placeholder for skip connection

        return layers


    def forward(self, bands, descriptors):

        batch_size,num_bands,descriptor_size = descriptors.shape
        reshaped_desc = descriptors.reshape(batch_size*num_bands,descriptor_size)
        for i,layer in enumerate(self.layers):
            new_reshaped_desc = layer(reshaped_desc)
            if self.skips and isinstance(layer,torch.nn.Identity) and new_reshaped_desc.shape == reshaped_desc.shape:
                new_reshaped_desc = new_reshaped_desc + reshaped_desc
            reshaped_desc = new_reshaped_desc

        return bands, reshaped_desc.reshape(batch_size,num_bands,self.layer_sizes[-1])

    def get_output_size(self):
        return self.layer_sizes[-1]


class PermutedFCLBlock(FCLBlock):
    """
    An FCL block that permutes the input descriptors before passing them through the 
    FCL layers. This leads to N^2 pairs of descriptors being passed through the FCL layers.
    """

    def __init__(self,config,input_descriptor_dims=None, device='cuda'):
        super(PermutedFCLBlock,self).__init__(config,2*input_descriptor_dims)
        self.device = device

    def permute_descriptors(self,descriptors):
        batch_size,num_bands,descriptor_size = descriptors.shape
        
        # Pair each descriptor with every other descriptor
        copies = descriptors.repeat(1,num_bands,1)
        copies_t = torch.repeat_interleave(descriptors,num_bands,dim=1)
        pairs = torch.cat([copies,copies_t],dim=2)

        # Run through FCL block
        return pairs.reshape(batch_size,num_bands*num_bands,descriptor_size*2)
    
    def pool_permuted_pairs(self,descriptors):
        batch_size,num_bands_squared,descriptor_size = descriptors.shape
        num_bands = int(np.sqrt(num_bands_squared))
        assert num_bands_squared == num_bands**2
        descriptors = descriptors.reshape(batch_size,num_bands,num_bands,descriptor_size)
        return descriptors.mean(dim=2)
    
    def forward(self,bands,descriptors):
        descriptors = self.permute_descriptors(descriptors)
        for layer in self.layers:
            descriptors = layer(descriptors)
        descriptors = self.pool_permuted_pairs(descriptors)
        return bands, descriptors

class AttentionBlock(torch.nn.Module):
    """
    A series of Transformer Encoders (multiheaded).
    """
    def __init__(self, config, input_descriptor_dims=None, device='cuda'):
        super(AttentionBlock, self).__init__()
        self.num_transformerencoders = config['num_transformerencoders']
        self.num_heads = config['num_heads']
        self.intermediate_size = config['intermediate_size']
        self.dims_per_head = config['dims_per_head']
        self.dropout = config['dropout']
        self.skips = config['skips']
        self.input_descriptor_dims = input_descriptor_dims
        self.device = device
        assert self.input_descriptor_dims % self.num_heads == 0, "Input descriptor dims must be divisible by number of heads"
        assert self.input_descriptor_dims==self.get_output_size(), "Input descriptor dims must be equal to num_heads*dims_per_head"

        # get layers
        self.layers = torch.nn.ModuleList(self._get_layers())
        for layer in self.layers:
            self._init_weights(layer)
    def _get_layers(self):
        layers = []
        for i in range(self.num_transformerencoders):
            layers += [torch.nn.TransformerEncoderLayer(
                            d_model=self.dims_per_head*self.num_heads,
                            nhead=self.num_heads, 
                            dim_feedforward=self.intermediate_size, 
                            dropout=self.dropout
            )]
        return layers

    def _init_weights(self,layer):
        if type(layer) == torch.nn.Linear:
            torch.nn.init.kaiming_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)

    def forward(self, bands, descriptors):
        
        # Run through layers
        for i,layer in enumerate(self.layers):
            new_descriptors = layer(descriptors)
            if self.skips:
                new_bands = new_descriptors + descriptors
            descriptors = new_descriptors
        return bands, descriptors

    def get_output_size(self):
        return self.dims_per_head*self.num_heads

class BandEmbedding(torch.nn.Module):
    """
    A way of intelligently embedding the band values into the output latent space of SEnSeI.

    This is done by learning a frequency, phase offset  and gain as functions of the descriptor
    vectors, and then using these to embed the band values with scaled sinusoidal functions. This 
    allows the model to (hypothetically) select where and how to encode each band in the latent space.
    """
    def __init__(self, config, input_descriptor_dims=None, device='cuda'):
        super(BandEmbedding,self).__init__()
        self.embedding_dims = config['embedding_dims']
        self.head_layer_sizes = config['head_layer_sizes']
        self.skips_heads = config['skips_heads']
        self.input_descriptor_dims = input_descriptor_dims
        self.device = device
        self.normalize = config['normalize']
        self.it = 0
        # Construct config for FCLs
        head_config = {
            "input_descriptor_dims": self.input_descriptor_dims,
            "layer_sizes": self.head_layer_sizes+[self.embedding_dims], # remember last layer 
            "skips": self.skips_heads
        }
        
        # Simple FCL feedforward networks to learn the embedding parameters
        self.frequency_head = FCLBlock(head_config,input_descriptor_dims=self.input_descriptor_dims)
        self.phase_offset_head = FCLBlock(head_config,input_descriptor_dims=self.input_descriptor_dims)
        self.gain_head = FCLBlock(head_config,input_descriptor_dims=self.input_descriptor_dims)

        if self.normalize:
            self.norm = torch.nn.BatchNorm2d(self.embedding_dims,momentum=0.05)

    def forward(self, bands, descriptors):
        # Calculate embedding parameters for all descriptors
        frequencies = self.frequency_head(None,descriptors)[1][:,:,:,None,None]
        phase_offsets = self.phase_offset_head(None,descriptors)[1][:,:,:,None,None]
        gains = self.gain_head(None,descriptors)[1][:,:,:,None,None]

        embeddings = torch.zeros(bands.shape[0],self.embedding_dims,bands.shape[-2],bands.shape[-1]).to(self.device)
        bands = bands[:,:,None,:,:] # add embedding dimension
        # Calculate embeddings in loop, slower but much more memory efficient
        for i in range(bands.shape[1]):
            embeddings += gains[:,i,...]*torch.sin(frequencies[:,i,...]*(bands[:,i,...] + phase_offsets[:,i,...]))
        if self.normalize:
            embeddings = self.norm(embeddings)
        return embeddings, descriptors


    def get_output_size(self):
        return self.embedding_dims

class BandMultiplication(torch.nn.Module):
    def __init__(self, config, input_descriptor_dims=None, device='cuda'):
        super(BandMultiplication,self).__init__()
        self.input_descriptor_dims = input_descriptor_dims
        self.device = device
        self.normalize = config['normalize']
        if self.normalize:
            self.embedding_dims = self.input_descriptor_dims
            self.norm = torch.nn.BatchNorm2d(self.embedding_dims,momentum=0.05)

    def forward(self, bands, descriptors):
        embeddings = torch.zeros(bands.shape[0],self.input_descriptor_dims,bands.shape[-2],bands.shape[-1]).to(self.device)
        bands = bands[:,:,None,:,:] # add embedding dimension
        descriptors = descriptors[:,:,:,None,None] # add embedding dimension
        # Calculate embeddings in loop, slower but much more memory efficient
        for i in range(bands.shape[1]):
            d = descriptors[:,i,...]
            embeddings += d*(bands[:,i,...]+0.5)
        if self.normalize:
            embeddings = self.norm(embeddings)

        return embeddings, torch.squeeze(descriptors)

    def get_output_size(self):
        return self.input_descriptor_dims


class RecoveryModule(torch.nn.Module):
    """
    Used to create additional loss term for SEnSeIv2 during training.

    Simply, it estimates the original reflectance values of each band
    given the combined embedding outputted by SEnSeIv2. This enforces
    that SEnSeIv2 accurately retains information about the original 
    bands it was given. 

    Number of layers should be small: we want SEnSeIv2 to have the 
    information in a fairly straightforward representation, so we don't 
    want to add too many nonlinearities to this module.
    """

    def __init__(self,config):
        super(RecoveryModule,self).__init__()
        self.preconcatenation_layer_sizes = config['preconcatenation_layer_sizes']
        self.postconcatenation_layer_sizes = config['postconcatenation_layer_sizes']
        self.sampling_rate = config['sampling_rate'] 

        self.preconcatenation_layers = self._get_preconcatenation_layers()
        self.postconcatenation_layers = self._get_postconcatenation_layers()

    def _get_preconcatenation_layers(self):
        layers = []
        for i,layer_size in enumerate(self.preconcatenation_layer_sizes):
                layers.append(torch.nn.ReLU())
                layers.append(torch.nn.LazyLinear(layer_size))
        return torch.nn.Sequential(*layers)

    def _get_postconcatenation_layers(self):
        layers = []
        for i,layer_size in enumerate(self.preconcatenation_layer_sizes):
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.LazyConv2d(layer_size,(1,1)))

        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.LazyConv2d(1, (1,1)))
        return torch.nn.Sequential(*layers)

    def forward(self, embedding, descriptors):
        # Note - in comments referring to shapes: 
        #       B refers to batch size
        #       N refers to number of bands
        #       C refers to number of feature channels in embedding/descriptors
        #       X,Y refer to spatial dimensions of embedding

        # process descriptors
        descriptors = self.preconcatenation_layers(descriptors)

        # Sample embedding
        embedding = embedding[:,:,::self.sampling_rate,::self.sampling_rate]

        # Shape: B,N,X,Y
        band_estimates = []

        for i in range(descriptors.shape[1]):
            extended_descriptors = descriptors[:,i,...][:,:,None,None]
            extended_descriptors = extended_descriptors.repeat(1,1,embedding.shape[-2],embedding.shape[-1])
            # Shape: B,C+C',X,Y
            band_concat = torch.concat((embedding,extended_descriptors),dim=1)
            band_estimates.append(self.postconcatenation_layers(band_concat))

        band_estimates = torch.concat(band_estimates,dim=1)
        return band_estimates

def FullyConnectedNN(input_descriptor_dims,output_classes,device='cuda'):
    """Simple pixelwise FCL network for benchmarking (uses 1x1 convs)"""

    nn = torch.nn.Sequential(
        torch.nn.Conv2d(input_descriptor_dims,256,1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(256,256,1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(256,output_classes,1)
    )

    def init_weights(m):
        if type(m) == torch.nn.Conv2d:
            torch.nn.init.kaiming_uniform_(m.weight)
            m.bias.data.fill_(0.01)        

    nn.apply(init_weights)

    return nn

def load_model(config,weights=None,device='cuda'):
    """
    Helper function to load model from config
    """
    if config.get('SEnSeIv2', False):
        # Make path to config absolute
        if not os.path.isabs(config['SEnSeIv2']):
            config['SEnSeIv2'] = os.path.join(os.path.dirname(__file__),'..', config['SEnSeIv2'])
        with open(config['SEnSeIv2']) as f:
            sensei_config = yaml.load(f, Loader=yaml.FullLoader)
        sensei_config['device'] = device
        senseiv2 = SEnSeIv2(sensei_config)
        NUM_CHANNELS = senseiv2.output_size
    else:
        senseiv2 = None
        NUM_CHANNELS = config['NUM_CHANNELS']
    
    if config.get('RECOVERY_MODULE', False):
        with open(os.path.join(os.path.dirname(__file__),'..',config['RECOVERY_MODULE'])) as f:
            recovery_module_config = yaml.load(f, Loader=yaml.FullLoader)
        recovery_module = RecoveryModule(recovery_module_config)
    else:
        recovery_module = None
    if config['MODEL_TYPE'] == 'Segformer':
        segmenter = SegformerForSemanticSegmentation.from_pretrained(
                                config['SEGFORMER_CONFIG'],
                                num_labels=config['CLASSES'],
                            )
        new_config = segmenter.config

        new_config.num_channels = NUM_CHANNELS
        new_config.num_labels = config['CLASSES']

        # Make new model to get a first layer with correct num. of channels
        new_model = SegformerForSemanticSegmentation(new_config)

        # Replace first layer of pretrained model
        segmenter.segformer.encoder.patch_embeddings[0] = new_model.segformer.encoder.patch_embeddings[0]

        # Output of model() needs to be specified as logits
        logits_called = True

    elif config['MODEL_TYPE'] == 'DeepLabv3+':
        segmenter = smp.DeepLabV3Plus(
            config['DEEPLAB_CONFIG'],
            encoder_weights='imagenet',
            upsampling=1, # We do our own upsampling in full model
            in_channels=NUM_CHANNELS,
            classes=config['CLASSES'],
            activation=None
        )
        logits_called = False # Output of model() is logits, no need to specify

    elif config['MODEL_TYPE'] == 'NN':
        segmenter = FullyConnectedNN(NUM_CHANNELS,config['CLASSES'],device=device)
        logits_called = False # Output of model() is logits, no need to specify
        
    model = FullModel(senseiv2,segmenter,recovery_module=recovery_module, logits_called=logits_called).to(device)

    # Load checkpoint if given
    if weights is not None:
        if isinstance(weights,str):
            if not os.path.isabs(weights):
                weights = os.path.join(os.path.dirname(__file__),'..', weights)
            model.load_state_dict(torch.load(weights),strict=False)
        elif isinstance(weights,dict):
            model.load_state_dict(weights,strict=False)
            
    return model


if __name__=='__main__':
    import yaml

    with open('config/test.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    senseiv2 = SEnSeIv2(config)

    # Test forward pass

    import numpy as np
    import torch

    batch_size = 2
    num_bands = 11
    num_descriptors = 11
    descriptor_size = 74

    bands = torch.from_numpy(np.random.rand(batch_size,num_bands,256,256).astype('float32'))

    descriptors = [[
        {'band_type':'TOA Reflectance','min_wavelength':0.4,'max_wavelength':0.5},
        {'band_type':'TOA Reflectance','min_wavelength':0.5,'max_wavelength':0.6},
        {'band_type':'TOA Reflectance','min_wavelength':0.6,'max_wavelength':0.7},
        {'band_type':'TOA Reflectance','min_wavelength':0.7,'max_wavelength':0.8},
        {'band_type':'TOA Reflectance','min_wavelength':0.8,'max_wavelength':0.9},
        {'band_type':'SAR VV'},
        {'band_type':'SAR VH'},
        {'band_type':'SAR Angle'},
        {'band_type':'DEM'},
        {'band_type':'fill'},
        {'band_type':'fill'}
    ]]*2

    embeddings = senseiv2([bands,descriptors])

    print(embeddings.shape)


    import torchsummary

    torchsummary.summary(senseiv2)
