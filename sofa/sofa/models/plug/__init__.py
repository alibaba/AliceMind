from .tokenization_plug import BertTokenizer
from .application_plug import (
    PlugClassificationModel,
    PalmModel,
    PlugModel,
    ModelPreparationWrapper,
    PlugType,
)
from .data_plug import (
    TaskTypeName,
    DataProcessor,
    OCNLIProcessor,
    CMNLIProcessor,
    AFQMCProcessor,
    CSLProcessor,
    TNEWSProcessor,
    IFLYTEKProcessor,
    SingleSentenceClassificaitonProcessor,
    SingleSentenceMutlipleClassificaitonProcessor,
    MultipleSentenceClassificaitonProcessor,
    MultipleSentenceMultipleClassificaitonProcessor,
    SequenceLabelingSentenceClassificaitonProcessor,
    configure_data,
    get_train_val_test_data_clean,
    data_preparation_nlu
)
from .data_palm import (
    GenerationProcessor,
    NlpccKbqaProcessor,
    data_preparation_nlg
)

from .configuration_plug import PlugNLUConfig, PlugNLGConfig

from .distributed_plug import DistributedPlug, DistributedPlugNLG

from .trainer_plug import TrainerPlug, TrainerPlugNLG

from .arguments_plug import  PlugArgs


PROCESSOR_MAPPING = {
    TaskTypeName.GENERATION: GenerationProcessor(),
    TaskTypeName.SINGLE_CLASSIFICATION: SingleSentenceClassificaitonProcessor(),
    TaskTypeName.PAIR_CLASSIFICATION: MultipleSentenceClassificaitonProcessor(),
    TaskTypeName.MULTILABEL_CLASSIFICATION: SingleSentenceMutlipleClassificaitonProcessor(),
    TaskTypeName.SEQUENCE_LABELING: SequenceLabelingSentenceClassificaitonProcessor(),
    TaskTypeName.KBQA: NlpccKbqaProcessor(),
}
