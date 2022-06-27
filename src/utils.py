# utils files
import numpy as np
import pandas as pd
import os

def load_data(path):
    if path.endswith(".csv"):
        data=pd.read_csv(path)
    else:
        data=pd.read_csv(path,sep="\t")
    return data
    # data\mrpc.csv
    
def writeout(model="",task="",dataset="",sim_score=0,execution_time=False,comments=""):
    """return a text file that will 
    store the result of each method"""
    model_name = "Results/"+str(model)+".txt"
    with open(model_name, "a+") as f:
        f.write(f"Model : {model} \n")
        f.write(f"Task: {task} \n")
        f.write(f"Dataset : {dataset} \n")
        f.write("\n")
        if task == "paraphrasing":
            f.write("positive_similarit {:.3f}".format(sim_score[0]))
            f.write("\n")
            f.write("negative_similarit {:.3f}".format(sim_score[1]))
            f.write("\n")
            f.write("difference_similarit {:.3f}".format(sim_score[2]))
            f.write("\n")
        else:
            n=1
            for i in range(len(sim_score)):
                f.write("{} Similarity Score for n={} is {:.3f}".format(model,n,sim_score[i]))
                n+=1
                f.write("\n")
                      
        if execution_time:
            f.write(f"{model} Model Execution Time for {task} task: {execution_time}")
            print("##################")
            print(f"Model testing on {task} task result successfully save at {model_name}")
            print("##################")
        f.write("\n")                
        f.write("##################")
        f.write("\n") 
        f.close()



def similarity_between_sent(sent1_encoded, sent2_encoded):
    """report the avg. cosine similarity score b.w two pairs of sentences"""    
    similarity_score = []
    for i in range(len(sent1_encoded)):
        similarity_score.append(cosine_similarity(
            sent1_encoded[i], sent2_encoded[i]))

    return np.mean(similarity_score)


def cosine_similarity(a, b):
    """
Takes 2 vectors a, b and returns the cosine similarity according 
    to the definition of the dot product
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)
