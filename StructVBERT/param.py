# coding=utf-8
# Copyleft 2019 project LXRT.

import argparse
import random
import logging

import numpy as np
import torch

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def get_optimizer(optim):
    # Bind the optimizer
    if optim == 'rms':
        # print("Optimizer: Using RMSProp")
        logger.info("Optimizer: Using RMSProp")
        optimizer = torch.optim.RMSprop
    elif optim == 'adam':
        # print("Optimizer: Using Adam")
        logger.info("Optimizer: Using Adam")
        optimizer = torch.optim.Adam
    elif optim == 'adamax':
        # print("Optimizer: Using Adamax")
        logger.info("Optimizer: Using Adamax")
        optimizer = torch.optim.Adamax
    elif optim == 'sgd':
        # print("Optimizer: sgd")
        logger.info("Optimizer: sgd")
        optimizer = torch.optim.SGD
    elif 'bert' in optim:
        optimizer = 'bert'      # The bert optimizer will be bind later.
    else:
        assert False, "Please add your optimizer %s in the list." % optim

    return optimizer


def parse_args():
    parser = argparse.ArgumentParser()

    # Data Splits
    parser.add_argument("--train", default='train')
    parser.add_argument("--valid", default='valid')
    parser.add_argument("--test", default=None)

    # Training Hyper-parameters
    parser.add_argument('--batchSize', dest='batch_size', type=int, default=256)
    parser.add_argument('--test_batch_size', dest='test_batch_size', type=int, default=950)
    parser.add_argument('--optim', default='bert')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=9595, help='random seed')
    parser.add_argument("--hidden_dropout", default=0.0, type=float)

    parser.add_argument("--fix_language_bert",
                        default=False,
                        action='store_true',
                        help='whether fix language bert while fine-tuning.')
    parser.add_argument("--one_stream",
                        default=False,
                        action='store_true',
                        help='whether to use the one_stream style.')
    parser.add_argument("--safer_fp16",
                        default=False,
                        action='store_true',
                        help='whether use safer fp16.')
    parser.add_argument("--debug_mode",
                        default=False,
                        action='store_true',
                        help='whether to debug mode')

    # Debugging
    parser.add_argument('--output', type=str, default='snap/test')
    parser.add_argument("--fast", action='store_const', default=False, const=True)
    parser.add_argument("--tiny", action='store_const', default=False, const=True)
    parser.add_argument("--tqdm", action='store_const', default=False, const=True)

    # Model Loading
    parser.add_argument('--load', type=str, default=None,
                        help='Load the model (usually the fine-tuned model).')
    parser.add_argument('--patial_load', type=str, default=None,
                        help='Load the patial model.')
    parser.add_argument('--loadLXMERT', dest='load_lxmert', type=str, default=None,
                        help='Load the pre-trained LXMERT model.')
    parser.add_argument('--loadLXMERTQA', dest='load_lxmert_qa', type=str, default=None,
                        help='Load the pre-trained LXMERT model with QA answer head.')
    parser.add_argument("--fromScratch", dest='from_scratch', action='store_const', default=False, const=True,
                        help='If none of the --load, --loadLXMERT, --loadLXMERTQA is set, '
                             'the model would be trained from scratch. If --fromScratch is'
                             ' not specified, the model would load BERT-pre-trained weights by'
                             ' default. ')
    parser.add_argument("--bert_model", default=None, type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument('--hidden_size', type=int, default=768)

    parser.add_argument('--bert_fix_no_str', type=str, default=None,
                        help='fix the no-matching of str in model when training.')
    parser.add_argument('--bert_fix_str', type=str, default=None,
                        help='fix the matching of str in model when training.')

    # Optimization
    parser.add_argument("--mceLoss", dest='mce_loss', action='store_const', default=False, const=True)

    # LXRT Model Config
    # Note: LXRT = L, X, R (three encoders), Transformer
    parser.add_argument("--llayers", default=0, type=int, help='Number of Language layers')
    parser.add_argument("--xlayers", default=12, type=int, help='Number of CROSS-modality layers.')
    parser.add_argument("--rlayers", default=0, type=int, help='Number of object Relationship layers.')

    parser.add_argument('--middle_layer', type=int, default=0, help='middle layer for match')

    # LXMERT Pre-training Config
    parser.add_argument("--taskMatched", dest='task_matched', action='store_const', default=False, const=True)
    parser.add_argument("--taskMaskLM", dest='task_mask_lm', action='store_const', default=False, const=True)
    parser.add_argument("--taskObjPredict", dest='task_obj_predict', action='store_const', default=False, const=True)
    parser.add_argument("--taskQA", dest='task_qa', action='store_const', default=False, const=True)
    parser.add_argument("--visualLosses", dest='visual_losses', default='obj,attr,feat', type=str)
    parser.add_argument("--qaSets", dest='qa_sets', default=None, type=str)
    parser.add_argument("--wordMaskRate", dest='word_mask_rate', default=0.15, type=float)
    parser.add_argument("--objMaskRate", dest='obj_mask_rate', default=0.15, type=float)

    # Training configuration
    parser.add_argument("--multiGPU", action='store_const', default=False, const=True)
    parser.add_argument("--num_workers", dest='num_workers', default=0, type=int)
    parser.add_argument("--amp_type", default=None, type=str, help="whether to use mix precision, must in [O0, O1, O2, O3]")

    parser.add_argument("--conditional_mask", dest='conditional_mask', action='store_const', default=False, const=True)
    parser.add_argument("--only_mask_words", dest='only_mask_words', action='store_const', default=False, const=True)
    parser.add_argument("--whole_word_mask", dest='whole_word_mask', action='store_const', default=False, const=True)
    parser.add_argument("--image_wwm", dest='image_wwm', action='store_const', default=False, const=True)
    parser.add_argument("--add_box_size", dest='add_box_size', action='store_const', default=False, const=True)
    parser.add_argument("--use_struct_emb", dest='use_struct_emb', action='store_const', default=False, const=True)
    parser.add_argument("--max_2d_position_embeddings", default=160, type=int, help='max struct position length')

    parser.add_argument("--use_hdf5", dest='use_hdf5', action='store_const', default=False, const=True)
    parser.add_argument("--use_npz", dest='use_npz', action='store_const', default=False, const=True)
    parser.add_argument("--use_jpg", dest='use_jpg', action='store_const', default=False, const=True)
    parser.add_argument("--image_hdf5_file", dest='image_hdf5_file', default='data/mscoco_imgfeat/train2014_obj36.tsv.token,data/mscoco_imgfeat/val2014_obj36.tsv.token', type=str)
    parser.add_argument("--padding", dest='padding', action='store_const', default=False, const=True)
    parser.add_argument("--with_score", dest='with_score', action='store_const', default=False, const=True)
    parser.add_argument("--clip_norm", dest='clip_norm', default=5., type=float)

    parser.add_argument("--merge_for_submit", dest='merge_for_submit', action='store_const', default=False, const=True)
    parser.add_argument('--merge_dir', type=str, default='snap/vqa/test_vqa_ensemble_predict_best_submission')

    parser.add_argument('--max_objects', type=int, default=100)
    parser.add_argument("--use_multi_view", dest='use_multi_view', action='store_const', default=False, const=True)

    parser.add_argument("--from_config_file",
                        default=False,
                        action='store_true',
                        help='whether initial config from config file.')
    parser.add_argument("--bert_config_file", default=None, type=str,
                        help="The config json file corresponding to the pre-trained BERT model. "
                             "This specifies the model architecture.")
    parser.add_argument('--single_image_file', type=str, default=None,
                        help='single image hdf5 file for pre-training image modality.')
    parser.add_argument("--revise_match_task",
                        default=False,
                        action='store_true',
                        help='whether to revise match task.')
    parser.add_argument("--prefetch",
                        default=False,
                        action='store_true',
                        help='whether to prefetch the data.')
    parser.add_argument("--paired_attn",
                        default=False,
                        action='store_true',
                        help='whether to use paired attn for NLVR task')
    parser.add_argument("--linear_nlvr",
                        default=False,
                        action='store_true',
                        help='whether to only use linear nlvr head')
    parser.add_argument("--only_use_relevant_dets",
                        default=False,
                        action='store_true',
                        help='whether to only_use_relevant_dets')

    parser.add_argument("--max_seq_length", default=40, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded. Must match data generation.")
    parser.add_argument("--nlvr_max_seq_length", default=20, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded. Must match data generation.")
    parser.add_argument("--concat_attention",
                        default=False,
                        action='store_true',
                        help='whether to use concat attention')
    parser.add_argument("--no_qa_data",
                        default=False,
                        action='store_true',
                        help='whether to remove qa data and task.')
    parser.add_argument("--use_vit",
                        default=False,
                        action='store_true',
                        help='vit mode.')
    parser.add_argument('--image_crop_size_h', type=int, default=448)
    parser.add_argument('--image_crop_size_w', type=int, default=448)
    parser.add_argument("--bin_h_num", type=int, default=8)
    parser.add_argument("--bin_w_num", type=int, default=8)
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")

    # distributed parameters
    parser.add_argument("--local_rank", type=int)
    parser.add_argument('--total_instances', type=int, default=8967075)
    parser.add_argument("--read_local_data",
                        default=False,
                        action='store_true',
                        help='whether to read image from local disk.')

    # Parse the arguments.
    args = parser.parse_args()

    # Bind optimizer class.
    args.optimizer = get_optimizer(args.optim)

    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    return args


args = parse_args()
