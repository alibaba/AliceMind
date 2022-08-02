
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
import torch
import argparse
from tqdm import tqdm
from transformers import BertTokenizerFast, BertForQuestionAnswering, AdamW

class squad_data(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.data.items()}

    def __len__(self):
        return len(self.data.input_ids)

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
        help="Input training JSON SQUAD V2.0 file",
    )
    # Optional parameters
    parser.add_argument("--epoch", default=3, type=int, help="Number of training epoches")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="Learning rate for Adam")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay for training")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size for training")
    parser.add_argument("--step_size", default=500, type=int, help="Step size for learning rate scheduler")
    parser.add_argument("--gamma", default=0.5, type=float, help="Multiplicative factor of learning rate decay")

    return parser.parse_args()

def parse_squad(path):
    with open(path, 'rb') as f:
        dataset = json.load(f)
    contexts = []
    questions = []
    answers = []
    for data in dataset['data']:
        for passage in data['paragraphs']:
            context = passage['context']
            for qas in passage['qas']:
                question = qas['question']
                for answer in qas['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)
    return contexts, questions, answers

def data_preprocess(encodings, answers, max_len):
    starts = []
    ends = []
    for answer in answers:
        start = answer['answer_start']
        end = start + len(answer['text'])
        answer['answer_end'] = end
    for i, answer in enumerate(answers):
        offset = 1
        starts.append(encodings.char_to_token(i, answer['answer_start']))
        ends.append(encodings.char_to_token(i, answer['answer_end']))
        if not starts[-1]: starts[-1] = max_len
        while not ends[-1]:
            ends[-1] = encodings.char_to_token(i, answer['answer_end'] - offset)
            offset += 1
    encodings.update({'start': starts, 'end': ends})

def train(args, train_dataset):
    model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")
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
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.epoch):
        losses = []
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start'].to(device)
            end_positions = batch['end'].to(device)
            loss = model(input_ids, attention_mask=attention_mask,
                            start_positions=start_positions,
                            end_positions=end_positions)[0]
            loss.backward()
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                losses.append(loss.item())
        print("Epoch: {} Loss: {}".format(epoch, sum(losses)/len(losses)))

if __name__ == "__main__":
    args = parse_args()
    train_contexts, train_questions, train_answers = parse_squad(args.train_file)
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
    data_preprocess(train_encodings, train_answers, tokenizer.model_max_length)
    train_dataset = squad_data(train_encodings)
    train(args, train_dataset)