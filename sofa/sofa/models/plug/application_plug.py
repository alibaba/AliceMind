from enum import Enum
from functools import reduce
from this import d
from .modeling_plug import BertForPreTraining, PalmForPreTraining, BertForClassification
from .configuration_plug import PlugNLUConfig, PlugNLGConfig
import torch
import os
from collections import OrderedDict
from sofa.utils import logging
from ctypes import CDLL, c_int, c_float, c_void_p
import errno
logger = logging.get_logger(__name__)

# from sofa.utils.modeling_utils import Application
class PlugType(Enum):
    NLU_ORIGIN = "plug_nlu_original-8192"
    NLG_ORIGIN = "plug_nlg_original-8192"

class PlugModel(torch.nn.Module):
    
    def __init__(self, config):
        super(PlugModel, self).__init__()
        self.model = BertForPreTraining(config)

    def forward(self, input_tokens, token_type_ids=None,
                attention_mask=None, checkpoint_activations=False):
        return self.model(
            input_tokens, token_type_ids, attention_mask,
            checkpoint_activations=checkpoint_activations)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.model.state_dict(destination=destination, prefix=prefix,
                                     keep_vars=keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        return self.model.load_state_dict(state_dict, strict=strict)

class PlugClassificationModel(torch.nn.Module):

    def __init__(self, config):
        super(PlugClassificationModel, self).__init__()
        self.model = BertForClassification(config, config.num_of_classes)

    def forward(self, input_tokens, token_type_ids=None,
                attention_mask=None, checkpoint_activations=False, detach_index=-1):
        return self.model(
            input_tokens, token_type_ids, attention_mask,
            checkpoint_activations=checkpoint_activations,
            detach_index=detach_index)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.model.state_dict(destination=destination, prefix=prefix,
                                     keep_vars=keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        return self.model.load_state_dict(state_dict, strict=strict)

class PalmModel(torch.nn.Module):
    
    def __init__(self, config):
        super(PalmModel, self).__init__()
        config.LR_weight_rank = [config.LR_weight_rank, config.LR_weight_alpha]
        config.LR_mask_rank = [config.LR_mask_rank, config.LR_mask_alpha]
        self.model = PalmForPreTraining(config)

    def forward(self, input_tokens, token_type_ids=None,
                attention_mask=None, target_tokens=None, position_ids=None, decode_attention_mask=None, checkpoint_activations=False, is_infer=False, sequence_output=None, parallel_output=True):
        return self.model(
            input_tokens, token_type_ids, attention_mask, target_tokens, position_ids, 
            decode_attention_mask, checkpoint_activations=checkpoint_activations, is_infer=is_infer, sequence_output=sequence_output, parallel_output=parallel_output)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.model.state_dict(destination=destination, prefix=prefix,
                                     keep_vars=keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        return self.model.load_state_dict(state_dict, strict=strict)

class ModelPreparationWrapper(object):
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.final_model_num = 1
    
    # model files 1 -> 8
    def _model_split(self, model, mp1, mp2):
        model_name = model
        model = torch.load(model)
        split = mp2 // mp1
        params = model['module']
        new_params = [{} for i in range(split)]
        for key in params:
            old_device = params[key].device.index
            new_device = old_device * split
            ## split in first dim
            # print (key)
            if key in ['bert.embeddings.word_embeddings.weight', \
                    'cls.predictions.decoder_weight', \
                    'cls.predictions.bias'] or 'intermediate' in key:
                split_param = params[key].split(params[key].size(0)//split, dim=0)
                for i in range(split):
                    new_params[i][key] = split_param[i].cuda((new_device + i) % 8)
            ## split in last dim
            elif 'output.dense.weight' in key:
                split_param = params[key].split(params[key].size(-1)//split, dim=-1)
                for i in range(split):
                    new_params[i][key] = split_param[i].cuda((new_device + i) % 8)
            elif 'output.weight' in key:
                split_param = params[key].split(params[key].size(-1)//split, dim=-1)
                for i in range(split):
                    new_params[i][key] = split_param[i].cuda((new_device + i) % 8)
            elif 'attention.dense.weight' in key:
                split_param = params[key].split(params[key].size(-1)//split, dim=-1)
                for i in range(split):
                    new_params[i][key] = split_param[i].cuda((new_device + i) % 8)
            ## split self attention
            elif 'query_key_value' in key:
                q, k, v = params[key].split(params[key].size(0)//3, dim=0)
                q = q.split(q.size(0)//split, dim=0)
                k = k.split(k.size(0)//split, dim=0)
                v = v.split(v.size(0)//split, dim=0)
                for i in range(split):
                    new_params[i][key] = torch.cat([q[i], k[i], v[i]], dim=0).cuda((new_device + i) % 8)
            elif '.key_value.' in key:
                k, v = params[key].split(params[key].size(0)//2, dim=0)
                k = k.split(k.size(0)//split, dim=0)
                v = v.split(v.size(0)//split, dim=0)
                for i in range(split):
                    new_params[i][key] = torch.cat([k[i], v[i]], dim=0).cuda((new_device + i) % 8)
            elif '.query.' in key:
                q = params[key]
                q = q.split(q.size(0)//split, dim=0)
                for i in range(split):
                    new_params[i][key] = q[i].cuda((new_device + i) % 8)
            else:
                for i in range(split):
                    new_params[i][key] = params[key].cuda((new_device + i) % 8)
        for i in range(split):
            del model['module']
            model['module'] = new_params[i]
            cnt = 0
            for key in new_params[i]:
                cnt += new_params[i][key].numel()
            # logger.info(cnt, new_device + i)
            #torch.save(model, 'mp_rank_{:02d}'.format(new_device + i) + '_model_states.pt')
            torch.save(model, self.model_dir+'/mp_rank_{:02d}'.format(new_device + i) + '_model_states.pt')

    # model files 8 -> 1
    def _model_concat(self, models, mp1, mp2, target_dir, skip=False):
        if skip:
            logger.info("skipped the concat steps")
            return models
        
        logger.info("Starting merging the models")
        
        new_models = []
        merge = mp1 // mp2 
        for i in range(0, mp1, merge):
            sub_models = models[i:i + merge]
            params = [x['module'] for x in sub_models]
            new_params = {}
            for key in params[0]:
                new_device = i // merge 
                # logger.info(key)
                ## merge in first dim
                if key in ['bert.embeddings.word_embeddings.weight', \
                        'cls.predictions.decoder_weight', \
                        'cls.predictions.bias'] or 'intermediate' in key:
                    merge_param = torch.cat([x[key] for x in params], dim=0)
                    new_params[key] = merge_param.cpu()
                    #new_params[key] = merge_param.cuda(new_device)
                ## split in last dim
                elif 'output.dense.weight' in key or 'output.weight' in key or 'attention.dense.weight' in key:
                    merge_param = torch.cat([x[key] for x in params], dim=-1)
                    new_params[key] = merge_param.cpu()
                    #new_params[key] = merge_param.cuda(new_device)
                ## _encoder_qkv_concat
                elif 'encoder' in key and '.query_key_value' in key:
                    new_params[key] = params[0][key]
                ## split self attention
                elif 'query_key_value' in key:
                    q_k_v = [x[key].split(x[key].size(0)//3, dim=0) for x in params]
                    qs = [x[0] for x in q_k_v]
                    ks = [x[1] for x in q_k_v]
                    vs = [x[2] for x in q_k_v]
                    q = torch.cat(qs, dim=0)
                    k = torch.cat(ks, dim=0)
                    v = torch.cat(vs, dim=0)
                    new_params[key] = torch.cat([q, k, v], dim=0).cpu()
                    #new_params[key] = torch.cat([q, k, v], dim=0).cuda(new_device)
                elif '.key_value.' in key:
                    k_v = [x[key].split(x[key].size(0)//2, dim=0) for x in params]
                    ks = [x[0] for x in k_v]
                    vs = [x[1] for x in k_v]
                    k = torch.cat(ks, dim=0)
                    v = torch.cat(vs, dim=0) 
                    new_params[key] = torch.cat([k, v], dim=0).cpu()
                    #new_params[key] = torch.cat([k, v], dim=0).cuda(new_device)
                elif '.query.' in key:
                    qs = [x[key] for x in params]
                    q = torch.cat(qs, dim=0)
                    new_params[key] = q.cpu()
                    #new_params[key] = q.cuda(new_device)
                else:
                    new_params[key] = params[0][key].cpu() 
                    #new_params[key] = params[0][key].cuda(new_device) 
            for model in sub_models:
                del model['module']
            sub_models[0]['module'] = new_params
            cnt = 0
            for key in new_params:
                cnt += new_params[key].numel()
            # logger.info(cnt, new_device)
            # torch.save(sub_models[0], target_dir+'/mp_rank_{:02d}'.format(new_device) + '_model_states.pt_merge')
            new_models.append(sub_models[0])

        logger.info("Generating merged model done")
        return new_models

        # if remove:
        #     self._remove_intermidate_models(models)
        #     logger.info("Removed the original {} models".format(mp1))
        
    # model qkv to q,k,v
    def _encoder_qkv_split(self, target_dir, n=8):
        all_data = []
        qkv_data = {}
        kv_data = {}
        split_data = {}
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)

        for i in range(n):
            model = self.model_dir+'/mp_rank_{:02d}_model_states.pt'.format(i)
            data = torch.load(model)
            all_data.append(data)

            for key in data['module']:
                if 'query_key_value' in key and 'encoder' in key:
                    weight = data['module'][key].detach().cpu()
                    q,k,v = torch.split(weight, weight.shape[0]//3, dim=0)
                    if key not in qkv_data:
                        qkv_data[key] = [(q, k, v)]
                    else:
                        qkv_data[key].append((q, k, v))
                    # logger.info("load", i, key, data['module'][key].shape)
            logger.info(i)

        for key in qkv_data:
            q = torch.cat([d[0] for d in qkv_data[key]], 0)
            k = torch.cat([d[1] for d in qkv_data[key]], 0)
            v = torch.cat([d[2] for d in qkv_data[key]], 0)
            split_data[key.replace("query_key_value", "query")] = q
            split_data[key.replace("query_key_value", "key")] = k
            split_data[key.replace("query_key_value", "value")] = v

        for i in range(n):
            data = all_data[i]
            
            for key in split_data:
                data['module'][key] = split_data[key][split_data[key].shape[0]//n*i:split_data[key].shape[0]//n*(i+1)].cuda(i) 
                # logger.info("save", i, key, data['module'][key].shape)

            for key in qkv_data:
                del data['module'][key]

            torch.save(data, target_dir+'/mp_rank_{:02d}_model_states.pt'.format(i))

    # model q,k,v to qkv
    def _encoder_qkv_concat(self, models, target_dir, n=8, skip=False):
        if skip:
            logger.info("skipped the qkv concat steps")
            return models
        
        logger.info("Start the qkv concat conversion")

        if not os.path.exists(target_dir):
            os.mkdir(target_dir)

        # qkv_data = {}
        # for i in range(n):
        #     data = models[i]

        #     for key in data['module']:
        #         if 'encoder' in key:
        #             if '.query' in key:
        #                 q = data['module'][key].detach().cpu()
        #                 k = data['module'][key.replace('.query', '.key')].detach().cpu()
        #                 v = data['module'][key.replace('.query', '.value')].detach().cpu()
        #                 weight = torch.cat([q, k, v], dim=0)
        #                 new_key = key.replace('.query', '.query_key_value')
        #                 qkv_data[new_key] = weight
        #                 # logger.info("load", i, key, data['module'][key].shape)

        #     for key in qkv_data:
        #         # data['module'][key] = qkv_data[key].cuda(i) 
        #         data['module'][key] = qkv_data[key]
        #         # logger.info("save", i, key, data['module'][key].shape)

        #     for key in qkv_data:
        #         del data['module'][key.replace('.query_key_value', '.query')]
        #         del data['module'][key.replace('.query_key_value', '.key')]
        #         del data['module'][key.replace('.query_key_value', '.value')]

        qkv_keys = []
        params = [x['module'] for x in models]
        for key in params[0]:
            if 'encoder' in key:
                if '.query' in key:
                    qkv_keys.append((key, key.replace('.query', '.key'), key.replace('.query', '.value')))
        for qkv_key in qkv_keys:
            q_key, k_key, v_key = qkv_key
            qs = [param[q_key].detach() for param in params]
            ks = [param[k_key].detach() for param in params]
            vs = [param[v_key].detach() for param in params]
            q = torch.cat(qs, dim=0)
            k = torch.cat(ks, dim=0)
            v = torch.cat(vs, dim=0)
            new_key = q_key.replace('.query', '.query_key_value')
            params[0][new_key] = torch.cat([q, k, v], dim=0)
            for param in params:
                del param[q_key]
                del param[k_key]
                del param[v_key]

        logger.info("Converted the qkv models")
        return models

    # dense model to sparse model
    def _convert_weights(self, target_dir):
        ckpt_list = os.listdir(self.model_dir)
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
            
        pruning_threshold=0.01

        for ckpt in ckpt_list:
            new_ckpt = OrderedDict()
            data = torch.load(f"{self.model_dir}/{ckpt}")

            old_ckpt = data['module']

            for key in old_ckpt:
                if key.endswith('.mask'):
                    old_ckpt[key.replace('.mask', '.weight')] = old_ckpt[key.replace('.mask', '.weight')] * old_ckpt[key]
                if key.endswith('.weight_mask'):
                    mask_scores = old_ckpt[key]
                    _, idx = mask_scores.flatten().sort(descending=True)
                    j = int(pruning_threshold * mask_scores.numel())
                    mask = mask_scores.clone()
                    flat_out = mask.flatten()
                    flat_out[idx[j:]] = 0
                    flat_out[idx[:j]] = 1
                    old_ckpt[key.replace('.weight_mask', '.weight')] = old_ckpt[key.replace('.weight_mask', '.weight')] * mask

                    # logger.info(key)

            for key in old_ckpt:
                if not (key.endswith('.mask') or key.endswith('.mask_scores') or key.endswith('.weight_mask')):
                    new_ckpt[key] = old_ckpt[key]
                    # logger.info(ckpt, key)

            data['module'] = new_ckpt
            torch.save(data, f"{target_dir}/{ckpt}")
    
    # clean up intermidate models
    def _remove_intermidate_models(self, models_dir):
        try:
            for model_dir in models_dir:
                os.remove(model_dir)
        except:
            logger.info("Error while deleting files ", models_dir)
    
    # convert pt file to bin file    
    def _convert_filetype(self, models, target_dir, infer_gpu_num=1, is_enc_sparse=True, is_dec_sparse=True, skip=False): 
        encoder_sparse_set = { "attention.self.query_key_value.weight",
                               "attention.output.dense.weight", 
                               "intermediate.dense.weight", 
                               "output.dense.weight" }
        decoder_sparse_set = { "attention.query_key_value.weight",
                               "attention.dense.weight",
                               "cross_attention.query.weight",
                               "cross_attention.key_value.weight",
                               "cross_attention.dense.weight",
                               "intermediate.weight",
                               "output.weight" }
        
        if skip:
            logger.info("skipped the model file conversion steps")
            return 
                
        logger.info("Starting to converted plug model files")

        for i in range(infer_gpu_num):
            saved_dir = target_dir + "/%d-gpu/" % infer_gpu_num
            if (os.path.exists(saved_dir) == False):
                os.makedirs(saved_dir)

            ckpt = models[i]['module']
            for key in ckpt:
                saved_file = saved_dir + key + "_{}.bin".format(i)
                # logger.info(saved_file)
                ckpt_np = ckpt[key].cpu().numpy()
                ckpt_np.tofile(saved_file)

                ## generate sparse cache files
                if is_enc_sparse and ".encoder" in key and key.endswith("weight") and reduce(lambda x, y: x or y, [i in key for i in encoder_sparse_set]):
                    m, n = ckpt[key].size()
                    libc = self.load_libc()
                    libc.NumpyToCache(ckpt_np.ctypes.data_as(c_void_p), m, n, saved_file.encode())
                if is_dec_sparse and ".decoder" in key and key.endswith("weight") and reduce(lambda x, y: x or y, [i in key for i in decoder_sparse_set]):
                    m, n = ckpt[key].size()
                    libc = self.load_libc()
                    libc.NumpyToCache(ckpt_np.ctypes.data_as(c_void_p), m, n, saved_file.encode())

        logger.info("converted plug model files")
    
    def load_libc(self, so_filename = "gpu_param.so", c_src_filename = "gpu_param.cpp"):
        so_path = os.path.dirname(__file__) + '/' + so_filename
        if os.path.exists(so_path):
            return CDLL(so_path) 
        c_src_path = os.path.dirname(__file__) + '/' + c_src_filename
        if not os.path.exists(c_src_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), c_src_filename)
        os.system(f"g++ -shared -fPIC {c_src_path} -o {so_path}")
        return CDLL(so_path)

    # convert finetune model to serving model
    def model_serving_preparation(self, target_dir, init_model_num=8, remove_origin=False):
        
        ### new finetuned model has splited qkv encoder
        ### new finetuned model has 8 models
        ### serving model need to be concated qkv encoder
        ### serving model need to be one model instead of 8
        
        # load models to cpu in advance for better performance
        models_path = [self.model_dir + '/mp_rank_{:02d}_model_states.pt'.format(i) for i in range(init_model_num)]
        models = [torch.load(model, map_location='cpu') for model in models_path]
        if remove_origin:
            self._remove_intermidate_models(models_path)    
            logger.info("Removed the {} qkv models".format(init_model_num))
        
        # convert qkv encoder
        models = self._encoder_qkv_concat(models, target_dir, init_model_num)

        # merge 8 to 1
        models = self._model_concat(models, init_model_num, self.final_model_num, target_dir)
        
        # ready for serving
        self._convert_filetype(models, target_dir)
    
        return target_dir
        