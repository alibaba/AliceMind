import torch
from utils import print_rank_0
from generate_plug import get_model_tokenizer
from generate_plug import generate_samples

class predict_plug(object):
    
    def init(self):
        print('==================== init ======================')
        model, tokenizer, args = \
            get_model_tokenizer('plug_model/vocab.txt', 'plug_model/28000')
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        print('================= init end =====================')

    def process(self, data):
        # data dict: {'passage': str, 'length': int}
        print_rank_0('================= process =================' + str(data))
        passage = data['passage']
        length = data['length']
        generate_passage = generate_samples(self.model, self.tokenizer, self.args, torch.cuda.current_device(), length, passage)
        return {'generage_passage': generate_passage}

if __name__ == '__main__':
    data = {'passage': '段誉轻挥折扇，摇了摇头，说', 'length': 512}
    plug = predict_plug()
    plug.init()
    print_rank_0(plug.process(data))
