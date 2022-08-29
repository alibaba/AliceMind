# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team and Alibaba-inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import glob
import torch
import argparse
from tqdm import tqdm
from transformers import BertTokenizerFast, BertForMultipleChoice, AdamW

def parse_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained models",
    )
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        required=True,
        help="Input RACE dataset",
    )
    # Optional parameters
    parser.add_argument("--epoch", default=3, type=int, help="Number of training epoches")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="Learning rate for Adam")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay for training")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size for training")
    parser.add_argument("--step_size", default=500, type=int, help="Step size for learning rate scheduler")
    parser.add_argument("--gamma", default=0.5, type=float, help="Multiplicative factor of learning rate decay")

    return parser.parse_args()

def convert_race(paths):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    input_ids, attention_mask, segment_id, labels = [], [], [], []
    for path in paths:
        filenames = glob.glob(path+"/*txt")
        for filename in filenames:
            with open(filename, 'r') as file:
                raw = json.load(file)
                for i in range(len(raw['answers'])):
                    cur_input_ids, cur_attention_mask, cur_segment_id = [], [], []
                    for option in range(len(raw['options'][i])):
                        encodings = tokenizer(raw['article'], raw['questions'][i]+raw['options'][i][option], truncation=True, padding='max_length', max_length=320)
                        cur_input_ids.append(encodings.input_ids)
                        cur_attention_mask.append(encodings.attention_mask)
                        cur_segment_id.append(encodings.token_type_ids)

                    labels.append(ord(raw['answers'][i]) - ord('A'))
                    input_ids.append(cur_input_ids)
                    attention_mask.append(cur_attention_mask)
                    segment_id.append(cur_segment_id)

    return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(segment_id), torch.tensor(labels)

def train(args):
    model = BertForMultipleChoice.from_pretrained("bert-base-uncased")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.train()
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    total_input_ids, total_attention_mask, total_segment_id, total_labels = convert_race([args.train_file+"/train/middle", args.train_file+"/train/high"])
    train_dataset = torch.utils.data.TensorDataset(total_input_ids, total_attention_mask, total_segment_id, total_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.epoch):
        losses = []
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            loss = model(
                input_ids=batch[0].to(device), 
                attention_mask=batch[1].to(device), 
                token_type_ids=batch[2].to(device), 
                labels=batch[3].to(device))[0]
            loss.backward()
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                losses.append(loss.item())
        print("Epoch: {} Loss: {}".format(epoch, sum(losses)/len(losses)))

if __name__ == "__main__":
    args = parse_args()
    train(args)