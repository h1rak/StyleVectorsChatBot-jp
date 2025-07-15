import os
import torch
import transformers
from tqdm import tqdm
import pickle
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
from transformers import BitsAndBytesConfig
import numpy as np

df = pd.read_table("/home/santa/chatbot/data/wrime-ver1.tsv")
df.columns = df.columns.str.strip()
#emotion_names = ['Joy', 'Sadness', 'Anticipation', 'Surprise', 'Anger', 'Fear', 'Disgust', 'Trust' ]
#df['readers_emotion_intensities'] = df.apply(lambda x: [x['Avg. Readers_' + name] for name in emotion_names], axis=1)
#is_target = df['readers_emotion_intensities'].map(lambda x: max(x) >= 2)
#df = df[is_target]

def wrime_get_single_emo(df, max_only=True, threshold=2):
    emotion_names = ['Joy', 'Sadness', 'Anticipation', 'Surprise', 'Anger', 'Fear', 'Disgust', 'Trust']
    selected = []
    for idx, row in df.iterrows():
        scores = [row['Avg. Readers_' + name] for name in emotion_names]
        if max_only:
            max_idx = int(np.argmax(scores))
            if scores[max_idx] == 0:
                continue
            selected.append({"text": row["Sentence"], "labels": [max_idx], "id": idx, "emo": emotion_names[max_idx],"Train/Dev/Test":row["Train/Dev/Test"]})
        else:
            pos = [i for i, s in enumerate(scores) if s >= threshold]
            if len(pos) == 1:
                selected.append({"text": row["Sentence"], "labels": [max_idx], "id": idx, "emo": emotion_names[max_idx],"Train/Dev/Test":row["Train/Dev/Test"]})
    return pd.DataFrame(selected)

def wrime_get_weighted_emo(df, threshold=1):
    emotion_names = ['Joy', 'Sadness', 'Anticipation', 'Surprise', 'Anger', 'Fear', 'Disgust', 'Trust']
    selected = []
    for idx, row in df.iterrows():
        scores = [row['Avg. Readers_' + name] for name in emotion_names]
        
        # スコアが0の感情は除外
        if all(score == 0 for score in scores):
            continue
        
        # 閾値以上のスコアがある感情を全て選択
        weighted_labels = {}
        for i, score in enumerate(scores):
            if score >= threshold:  # 閾値以上のスコアだけ考慮
                weighted_labels[emotion_names[i]] = score
                #weighted_labels.append((emotion_names[i], score))  # 感情名とスコアのペアとして保存

        if weighted_labels:
            selected.append({
                "text": row["Sentence"],
                "labels": weighted_labels,  # スコアと感情のペアをラベルとして保存
                "id": idx,
                "Train/Dev/Test": row["Train/Dev/Test"]
            })
    return pd.DataFrame(selected)


df = wrime_get_weighted_emo(df)
print(df)
load_dotenv()

MODEL_PATH = os.getenv("MODEL_WEIGHTS_FOLDER")

PATH_TO_ACTIVATION_STORAGE = os.getenv("ACTIVATIONS_PATH")
Path(PATH_TO_ACTIVATION_STORAGE).mkdir(parents=True,exist_ok=True)

DEVICE = torch.device("cuda:0")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# 元のコードを以下に置き換え
model = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map="auto",  # .to(DEVICE)の代わりに使用
    torch_dtype=torch.float16
)

tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)
model.to(DEVICE)

def process_dataset(df):
    df = df
    actis,actis_train,actis_test = [],[],[]
    i = 0
    j = 0
    for index,row in tqdm(df.iterrows(),total=df.shape[0]):
        sentence = []
        sentence = row["text"].replace("\n","")
        input_tokens = tokenizer(sentence,return_tensors="pt").to(DEVICE)

        if len(input_tokens.input_ids) > 300:
            continue
        gen_text = model.forward(input_tokens.input_ids,output_hidden_states=True,return_dict=True)
        hidden_states = []

        for layer in gen_text["hidden_states"]:
            hidden_states.append(layer[0][-1].detach().cpu().numpy())
        if row["Train/Dev/Test"] == "train":
            actis_train.append([index,row,hidden_states])
        elif row["Train/Dev/Test"] == "test":
            actis_test.append([index,row,hidden_states])

        i += 1
    with open(f'{PATH_TO_ACTIVATION_STORAGE}/wieghted_activations_train.pkl', 'wb') as f:
        pickle.dump(actis_train, f)
    with open(f'{PATH_TO_ACTIVATION_STORAGE}/wieghted_activations_test.pkl', 'wb') as f:
        pickle.dump(actis_test, f)

    del actis
    del hidden_states

process_dataset(df)