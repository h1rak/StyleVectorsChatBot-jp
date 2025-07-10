import os
import torch
import numpy as np
from torch import nn
from transformers import pipeline
from tqdm import tqdm
from pathlib import Path

def calculate_means(train,test,labels,insertion_layers):
    means,total_mean,ovr_r_means = [],[],[]
    train,test = concat_layers(train,test,insertion_layers)
    for label in labels:
        label_samples = [entry[2] for entry in train if entry[1]["labels"][0] == label]
        label_samples += [entry[2] for entry in test if entry[1]["labels"][0] == label]
        r_labels = [entry[2] for entry in train if entry[1]['labels'][0] != label]
        r_labels += [entry[2] for entry in test if entry[1]['labels'][0] != label]

        means.append(np.mean(label_samples,0))
        ovr_r_means.append(np.mean(r_labels,0))

    total_mean.append(np.mean(means,0))

    return means,ovr_r_means,total_mean

def concat_layers(train,test,insertion_layers):
    for idx, entry in enumerate(train):
        concatenated_layers = np.concatenate([entry[2][i + 1] for i in insertion_layers]) 
        train[idx][2] = concatenated_layers
    for idx, entry in enumerate(test):
        concatenated_layers = np.concatenate([entry[2][i + 1] for i in insertion_layers])
        test[idx][2] = concatenated_layers        
    return train, test


def interactive_eval(emotions,means,ovr_r_means,total_mean,llm_model, tokenizer, insertion_layers, save_path, device):
    while True:
        emotion = input("['Joy', 'Sadness', 'Anticipation', 'Surprise', 'Anger', 'Fear', 'Disgust', 'Trust']から生成したい感情を指定してください：")
        lam = float(input("感情の強さを0~2で入力してください："))
        prompt = input("入力：")
        if prompt.strip() == "": break
        input_text = (
            "以下はタスクを説明する指示文です。"
            "この指示に対して適切な回答を作成してください。\r\n\r\n"
            f"### 指示:\r\n{prompt}\r\n\r\n### 回答:"
        )
        chat = tokenizer.apply_chat_template([{"role":"user","content":input_text}],tokenize=False)
        inputs = tokenizer(chat,return_tensors="pt").to(device)

        emo_idx = emotions.index(emotion)
        vec_split = np.split(means[emo_idx] - ovr_r_means[emo_idx],len(insertion_layers))
        for n,_ in enumerate(insertion_layers):
            llm_model.model.layers[insertion_layers[n]].mlp.steering_vector = nn.Parameter(torch.from_numpy(vec_split[n]).to(device))
            llm_model.model.layers[insertion_layers[n]].mlp.b = lam

        gen_tokens = llm_model.generate(inputs.input_ids,max_length=150)
        print(f"テキスト：{prompt} \n 感情:{emotion} \n lambda:{lam}")
        output = tokenizer.batch_decode(gen_tokens)[0].replace(input_text,"").replace("\n"," ").replace(";","-")
        print(f"生成文：{output}")

        
