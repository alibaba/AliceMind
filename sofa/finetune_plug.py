import os
import torch

os.environ["SOFA_BACKEND"] = 'huggingface'
from sofa.models.plug import (
    PlugArgs, 
    BertTokenizer, 
    PalmModel,
    DistributedPlugNLG,
    TrainerPlugNLG,
    PlugNLGConfig,
    data_preparation_nlg,
    PROCESSOR_MAPPING
)

from sofa.models.plug.data_palm import WeatherProcessor

from sofa.utils import mpu, print_rank_0

def processor_factory(task_type):
    if task_type in PROCESSOR_MAPPING.keys():
        return PROCESSOR_MAPPING[task_type]
    else:
        raise RuntimeError(f"The task type is not matched the system required")


if __name__ == "__main__":
    # get arguments
    args_reader = PlugArgs()
    args = args_reader.get_args()

    # get tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.load)

    # get model config
    config = PlugNLGConfig.from_json_file(args.load + '/config.json')
    args = args_reader.merge_config(config)

    # init distributed plug and tokenzier
    distributed_model = DistributedPlugNLG(tokenizer=tokenizer, args=args)
    tokenizer, config.vocab_size, config.type_vocab_size = distributed_model.setup_tokenizer_stuff(args)

    # get datasets
    # processor = processor_factory(args.task_type)
    processor = WeatherProcessor()
    train_dataset, eval_dataset, test_dataset, label_list, args = data_preparation_nlg(tokenizer=tokenizer, processor=processor, args=args)

    # get distributed model, optimizer, lr_scheduler, tokenizer
    model = PalmModel(config)
    distributed_model.set_model(model)
    model, optimizer, lr_scheduler, pruner = distributed_model.setup_model_and_optimizer()

    # get trainer 
    trainer = TrainerPlugNLG(
        model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
        args=args, train_dataset=train_dataset, pruner=pruner,
        eval_dataset=eval_dataset, test_dataset=test_dataset, tokenizer=tokenizer
    )
    
    # train the model
    print_rank_0("start training~~~~~~")
    trainer.train()
    
