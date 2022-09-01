import os
import sys
import json

def weather():
    os.system("wget https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/PALM/weather_train.txt \
        && mv weather_train.txt train.txt \
        && wget https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/PALM/weather_dev.txt \
        && mv weather_dev.txt dev.txt \
        ")
    
def dureaderqg():
    os.system("wget --no-check-certificate https://bj.bcebos.com/paddlenlp/datasets/DuReaderQG/train.json \
        && wget --no-check-certificate https://bj.bcebos.com/paddlenlp/datasets/DuReaderQG/dev.json \
        ")
    def json2txt(prefix: str):
        with open(prefix + ".json", 'r') as f_in:
            with open(prefix + ".txt", 'w') as f_out:
                for i, line in enumerate(f_in.readlines()):
                    s = json.loads(line)
                    if i > 0:
                        f_out.write('\n')
                    f_out.write(s["answer"] + "[SEP]" + s["context"] + '\t' + s["question"])
    process(json2txt)

def dureader_robust():
    os.system("wget --no-check-certificate https://dataset-bj.cdn.bcebos.com/dureader_robust/data/dureader_robust-data.tar.gz \
        && tar -zxvf dureader_robust-data.tar.gz \
        && rm -f dureader_robust-data.tar.gz \
        && mv dureader_robust-data/train.json ./ \
        && mv dureader_robust-data/dev.json ./ \
        && rm -rf dureader_robust-data \
        ")
    def json2txt(prefix: str):
        with open(prefix + ".json", 'r') as f_in:
            with open(prefix + ".txt", 'w') as f_out:
                paras = json.loads(f_in.read())["data"][0]["paragraphs"]
                for i, p in enumerate(paras):
                    if i > 0:
                        f_out.write('\n')
                    f_out.write(p["qas"][0]["answers"][0]["text"] + "[SEP]" + p["context"] + '\t' + p["qas"][0]["question"])
    process(json2txt)

def lcsts():
    os.system("wget --no-check-certificate https://bj.bcebos.com/paddlenlp/datasets/LCSTS_new/train.json \
        && wget --no-check-certificate https://bj.bcebos.com/paddlenlp/datasets/LCSTS_new/dev.json \
        ")
    def json2txt(prefix: str):
        with open(prefix + ".json", 'r') as f_in:
            with open(prefix + ".txt", 'w') as f_out:
                for i, line in enumerate(f_in.readlines()):
                    s = json.loads(line)
                    if i > 0:
                        f_out.write('\n')
                    f_out.write(s["content"] + '\t' + s["summary"])
    process(json2txt)

def process(fn):
    fn("train")
    fn("dev")
    os.remove("train.json")
    os.remove("dev.json")

if __name__ == "__main__":
    fn_map = {"weather": weather, "dureaderqg": dureaderqg, "dureader_robust": dureader_robust, "lcsts": lcsts}
    data_dir = sys.argv[1]
    data_type = sys.argv[2]
    print(f"Downloading {data_type} dataset to {data_dir}")
    os.chdir(f"{data_dir}")
    fn_map[data_type]()