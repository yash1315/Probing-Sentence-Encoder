import os
import sys
import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
import numpy as np
import json
# from src.utils import *
import warnings
sys.path.insert(0,"models\InferSent")
from models import InferSent
import torch
from laserembeddings import Laser
from sentence_transformers import SentenceTransformer
from gensim.models.doc2vec import Doc2Vec,TaggedDocument
import logging
import urllib.request
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')


warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# import argparser # add arguments for commandline later.
with open("src/config.json") as config_json:
    config = json.load(config_json)



def embed(input, model, encoder_name=""):
    if encoder_name == "use":
        return model(input)
    if encoder_name == "sbert":
        return [model.encode(i) for i in input]
    if encoder_name == "laser":
        return model.embeddings(input)
    if encoder_name == "inferSent":
        return model.encode(input, tokenize=True)
# intializing the Sentence encoders


def USE():
    logging.info("Loading USE Model")
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    use_model = hub.load(module_url)
    print("USE model loaded successfully....")
    return use_model

def SentBert(task=""):
    logging.info("Loading Sentence-Bert(SBERT) model")
    if task == "paraphrasing":
        model_sbert = SentenceTransformer(
            config["model_task"]["SBert"]["paraphrasing"])
    if (task == "synonym" or task == "antonym" or task == "jumbling"):
        model_sbert = SentenceTransformer(
            config["model_task"]["SBert"]["other"])
    return model_sbert

def download_file(url, dest):
    sys.stdout.flush()
    urllib.request.urlretrieve(url, dest)

def download_models(output_dir):
    logger.info('Downloading models into {}'.format(output_dir))

    download_file('https://dl.fbaipublicfiles.com/laser/models/93langs.fcodes',
                  os.path.join(output_dir, '93langs.fcodes'))
    download_file('https://dl.fbaipublicfiles.com/laser/models/93langs.fvocab',
                  os.path.join(output_dir, '93langs.fvocab'))
    download_file(
        'https://dl.fbaipublicfiles.com/laser/models/bilstm.93langs.2018-12-26.pt',
        os.path.join(output_dir, 'bilstm.93langs.2018-12-26.pt'))
    
class LASERMODEL:
    def __init__(self,output_dir):
        logging.info("Loading LASER models ")
        download_models("models/Laser")
        DEFAULT_BPE_CODES_FILE = os.path.join(output_dir, '93langs.fcodes')
        DEFAULT_BPE_VOCAB_FILE = os.path.join(output_dir, '93langs.fvocab')
        DEFAULT_ENCODER_FILE = os.path.join(output_dir,
                                            'bilstm.93langs.2018-12-26.pt')
        self.embedding_model = Laser(DEFAULT_BPE_CODES_FILE, DEFAULT_BPE_VOCAB_FILE, DEFAULT_ENCODER_FILE)
    def embeddings(self,x):
        return self.embedding_model.embed_sentences(x,lang='en')

def infersent():    
    logging.info("Loading InferSent Model")
    model_path = config["model_config"]["infersent"]["encoder"]["v2"]
    model = InferSent(config["model_config"]["infersent"]["params_model"])
    model.load_state_dict(torch.load(model_path))
    glove_embedding = config["model_config"]["infersent"]["model_version"]
    model.set_w2v_path(glove_embedding)
    model.build_vocab_k_words(config["model_config"]["infersent"]["vocab_size"])
    return model

def d2v(sent):
    """Doc2Vec model"""
    tokens = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(sent)]
    model = Doc2Vec(vector_size=20,window=5,min_count=1,workers=2,dm=1,epochs=20)
    model.build_vocab(tokens)
    return tokens,model


# if __name__ == "__main__":

    # ex1 = ["This is testing."]
    # temp = SentBert("paraphrasing")
    # temp2 = SentBert("synonym")
    # download_models("models/Laser/test")
    # temp=Laser()
    # temp=LASERMODEL(config["model_config"]["laser"]["path"])
    # temp = infersent()
    # temp,mod = d2v(ex1)
    # t1 = embed(ex1,temp,"infersent")
    # print("model_loaded")
