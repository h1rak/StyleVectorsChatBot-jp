import os
import torch
import numpy as np
from torch import nn
from transformers import pipeline
from tqdm import tqdm
from pathlib import Path
from torch.cuda.amp import autocast

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

def calculate_weighted_means(train, test, labels, insertion_layers):
    means, total_mean, ovr_r_means = [], [], []
    train, test = concat_layers(train, test, insertion_layers)

    for label in labels:
        label_samples = []
        label_weights = []  # 重みを保持するリスト

        # 'label'に対応するサンプルを集め、感情スコアを重みとして利用
        for entry in train + test:
            if label in entry[1]["labels"]:
                label_samples.append(entry[2])
                # 感情スコアを1〜3の範囲で重みに変換
                score = entry[1]["labels"][label]
                if score == 3:
                    weight = 1  # スコアが3なら重み1
                elif score == 2:
                    weight = 0.66  # スコアが2なら重み0.66
                else:
                    weight = 0.33  # スコアが1なら重み0.33
                label_weights.append(weight)

        # 他のラベル（反対の感情）に対しても同様に計算
        r_labels = []
        r_weights = []

        for entry in train + test:
            if label not in entry[1]['labels']:
                r_labels.append(entry[2])

        # 加重平均を計算するため、リストをNumPy配列に変換
        label_samples = np.array(label_samples, dtype=np.float32)
        r_labels = np.array(r_labels, dtype=np.float32)

        # 重みをNumPy配列に変換
        label_weights = np.array(label_weights, dtype=np.float32)

        # 各サンプルの重みで加重平均を計算
        weighted_mean = np.average(label_samples, axis=0, weights=label_weights)
        weighted_r_mean = np.average(r_labels, axis=0)

        means.append(weighted_mean)
        ovr_r_means.append(weighted_r_mean)

    total_mean.append(np.mean(means, 0))

    return means, ovr_r_means, total_mean

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
            "あなたは対話するキャラクターです。"
            "このユーザーからの入力に対して適切な回答を作成してください。\r\n\r\n"
            f"### 入力:\r\n{prompt}\r\n\r\n### 回答:"
        )
        chat = tokenizer.apply_chat_template([{"role":"user","content":input_text}],tokenize=False,add_special_tokens=False)
        inputs = tokenizer(chat,return_tensors="pt").to(device)

        emo_idx = emotions.index(emotion)
        vec_split = np.split(means[emo_idx] - ovr_r_means[emo_idx],len(insertion_layers))
        for n,_ in enumerate(insertion_layers):
            llm_model.model.layers[insertion_layers[n]].mlp.steering_vector = nn.Parameter(torch.from_numpy(vec_split[n]).to(device))
            llm_model.model.layers[insertion_layers[n]].mlp.b = lam

        gen_tokens = llm_model.generate(inputs.input_ids,max_length=300)
        print(f"テキスト：{prompt} \n 感情:{emotion} \n lambda:{lam}")
        output = tokenizer.batch_decode(gen_tokens)[0].replace(input_text,"").replace("\n"," ").replace(";","-")
        print(f"生成文：{output}")

        
def interactive_eval_autocast(emotions, means, ovr_r_means, total_mean, llm_model, tokenizer, insertion_layers, save_path, device):
    while True:
        emotion = input("['Joy', 'Sadness', 'Anticipation', 'Surprise', 'Anger', 'Fear', 'Disgust', 'Trust']から生成したい感情を指定してください：")
        lam = float(input("感情の強さを0~2で入力してください："))
        prompt = input("入力：")
        if prompt.strip() == "":
            break
        input_text = (
            "あなたは対話するキャラクターです。"
            "このユーザーからの入力に対して適切な回答を作成してください。\r\n\r\n"
            f"### 入力:\r\n{prompt}\r\n\r\n### 回答:"
        )
        chat = tokenizer.apply_chat_template([{"role": "user", "content": input_text}], tokenize=False, add_special_tokens=False)
        inputs = tokenizer(chat, return_tensors="pt").to(device)

        emo_idx = emotions.index(emotion)
        vec_split = np.split(means[emo_idx] - ovr_r_means[emo_idx], len(insertion_layers))
        for n, _ in enumerate(insertion_layers):
            llm_model.model.layers[insertion_layers[n]].mlp.steering_vector = nn.Parameter(torch.from_numpy(vec_split[n]).to(device))
            llm_model.model.layers[insertion_layers[n]].mlp.b = lam

        # 推論部分をautocastでラップ
        with torch.amp.autocast('cuda', dtype=torch.float16):
            # 推論処理
            gen_tokens = llm_model.generate(inputs.input_ids, max_length=300)

        print(f"テキスト：{prompt} \n 感情:{emotion} \n lambda:{lam}")
        output = tokenizer.batch_decode(gen_tokens)[0].replace(input_text, "").replace("\n", " ").replace(";", "-")
        print(f"生成文：{output}")