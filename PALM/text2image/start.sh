git clone https://github.com/openai/CLIP
git clone https://github.com/alibaba/AliceMind.git
git clone https://github.com/CompVis/taming-transformers.git
pip install ftfy regex tqdm omegaconf pytorch-lightning
pip install kornia
pip install imageio-ffmpeg
pip install einops
mkdir steps


wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1mSp-4KfBwGKUAdWiW-ctOR9Qgi0a-w9B' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1mSp-4KfBwGKUAdWiW-ctOR9Qgi0a-w9B" -O palm_model_and_data.tar.gz && rm -rf /tmp/cookies.txt)
tar -zxvf palm_model_and_data.tar.gz

#下载数据集
curl -L -o vqgan_imagenet_f16_16384.yaml -C - 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1' #ImageNet 16384
curl -L -o vqgan_imagenet_f16_16384.ckpt -C - 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1' #ImageNet 16384

python train.py