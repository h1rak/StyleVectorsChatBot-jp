import os 
import pickle
import torch
import glob
from tqdm import tqdm
import random
from dotenv import load_dotenv
from utils.llm_model_utils import load_llm_model_with_insertions
from steering_helper import interactive_eval, calculate_means, calculate_weighted_means,interactive_eval_autocast
#from utils.steering_vector_loader import load_activations


load_dotenv()

def load_activations(vector_path):
    with open(f'{vector_path}/wieghted_activations_train.pkl','rb') as f:
        train = pickle.load(f)

    with open(f'{vector_path}/wieghted_activations_test.pkl','rb') as f:
        test = pickle.load(f)

    return train,test


DEVICE = torch.device("cuda:0")
INSERTION_LAYERS = [38,39,40]
model,tokenizer = load_llm_model_with_insertions(DEVICE,INSERTION_LAYERS)

ACTIVATION_VECTOR_PATH = os.getenv("ACTIVATIONS_PATH")
SAVE_PATH = os.getcwd()
train,test = load_activations(ACTIVATION_VECTOR_PATH)

train = [entry for entry in train if len(entry)==3]
test = [entry for entry in test if len(entry)==3]

emotion_names = ['Joy', 'Sadness', 'Anticipation', 'Surprise', 'Anger', 'Fear', 'Disgust', 'Trust']
emotion_idx = [0,1,2,3,4,5,6,7]

#means, ovr_r_means, total_mean = calculate_means(train, test, emotion_idx, INSERTION_LAYERS)
means, ovr_r_means, total_mean = calculate_weighted_means(train, test, emotion_names, INSERTION_LAYERS)
#interactive_eval(emotion_names, means, ovr_r_means, total_mean, llm_model=model, tokenizer=tokenizer, insertion_layers=INSERTION_LAYERS, save_path=SAVE_PATH, device=DEVICE)
interactive_eval_autocast(emotion_names, means, ovr_r_means, total_mean, llm_model=model, tokenizer=tokenizer, insertion_layers=INSERTION_LAYERS, save_path=SAVE_PATH, device=DEVICE)