from src.utils import *
from src.sentence_encoders import *
from src.word_replacer import *
import json
import random
import time
random.seed(0)
with open("src/config.json") as config_json:
    config = json.load(config_json)


def paraphrasing_task(encoder_model, encoder_model_name="", dataset_name="", sent1="", sent2="", sent3="", sent4=""):
    """
    Input: 
    encoder_model= [use, laser, infersent, sbert]: pass the model
    encoder_model_name= pass the name of the model in the string format like ["use", "sbert", "inferSent","laser"]
    dataset_name=either "mrpc","qqp", or "paws-wiki. 
    sent1 = First sentence with positive label
    sent2 = second sentence with positive label
    sent3 = first sentence with negative label
    sent4 = second sentence with negative label

    output: cosine similarity score, saved in a file.
    """
    start = time.time()
    embed_positive_sent_1 = embed(sent1, encoder_model, encoder_model_name)
    embed_positive_sent_2 = embed(sent2, encoder_model, encoder_model_name)
    embed_negative_sent_1 = embed(sent3, encoder_model, encoder_model_name)
    embed_negative_sent_2 = embed(sent4, encoder_model, encoder_model_name)
    positive_similarity = similarity_between_sent(
        embed_positive_sent_1, embed_positive_sent_2)
    negative_similarity = similarity_between_sent(
        embed_negative_sent_1, embed_negative_sent_2)
    difference_in_similarity = positive_similarity - negative_similarity
    end = time.time()
    et = end-start

    writeout(model=encoder_model_name, task="paraphrasing", dataset=dataset_name, sim_score=[
             positive_similarity, negative_similarity, difference_in_similarity], execution_time=et)


def synonym_and_antonym_replacement(encoder_model="", encoder_model_name="", task="", dataset_name="", original_sentence="", n=1):
    """
    Input: 
    encoder_model= [use, laser, infersent, sbert]: pass the model
    encoder_model_name= pass the name of the model in the string format like ["use", "sbert", "inferSent","laser"]
    task: either "synonyms" or "antonyms" -> str (all in lower case)
    dataset_name=either "mrpc","qqp", or "paws-wiki. 
    original_sentence= pass the sentences for which you want synonym/antonym replacement.
    n=1 is default, but you choose any natural number. it represent the number of times you want to replace words in a sentence.

    output: avg. cosine similarity score between original and synonym/antonym sentence, saved in a file.
    """
    start = time.time()
    data = {"original_sentence": original_sentence}
    for idx in range(n):
        data[f"n={idx+1}"] = [replace.sentence_replacement(
            i, n=idx+1, types=task) for i in data["original_sentence"]]

    df = pd.DataFrame(data, columns=[i for i in data.keys()])
    # saving data in csv file
    temp = config["save_perturbed_data"]["path"] + \
        "/"+dataset_name+f"_{task}_sent_pertub.csv"
    if os.path.exists(temp):
        pass
    else:
        df.to_csv(config["save_perturbed_data"]["path"] +
                  "/"+dataset_name+f"_{task}_sent_pertub.csv")
        print("Perturbed sentences generation saved at -> " +
              config["save_perturbed_data"]["path"]+"/"+dataset_name+f"_{task}_sent_pertub.csv")

    embeddings = {"embed_original": embed(
        df["original_sentence"], encoder_model, encoder_model_name)}
    for idx in range(n):
        embeddings[f"n={idx+1}"] = embed(df[f"n={idx+1}"],
                                         encoder_model, encoder_model_name)

    similarity_score = {}
    for idx in range(n):
        similarity_score[f"original_sentence-n={idx+1}"] = similarity_between_sent(
            embeddings["embed_original"], embeddings[f"n={idx+1}"])

    end = time.time()
    exec_time = end-start

    writeout(model=encoder_model_name, task=task, dataset=dataset_name,
             sim_score=list(similarity_score.values()), execution_time=exec_time)


def sentence_jumbling(encoder_model="", encoder_model_name="", dataset_name="", original_sentence="", n=1):
    """
    Input: 
    encoder_model= [use, laser, infersent, sbert]: pass the model
    encoder_model_name= pass the name of the model in the string format like ["use", "sbert", "inferSent","laser"]
    dataset_name=either "mrpc","qqp", or "paws-wiki. 
    original_sentence= pass the sentences for which you want synonym/antonym replacement.
    n=1 is default, but you choose any natural number. it represent the number of times you want to replace words in a sentence.

    output: avg. cosine similarity score between original and jumbled sentence, saved in a file.
"""
    start = time.time()
    data = {"original_sentence": original_sentence}

    for idx in range(n):
        data[f"n={idx+1}"] = [jumbler.random_swap(i, n=idx+1)
                              for i in data["original_sentence"]]

    df1 = pd.DataFrame(data, columns=[i for i in data.keys()])
    temp1 = config["save_perturbed_data"]["path"] + \
        "/"+dataset_name+"sentence_jumbling.csv"
    if os.path.exists(temp1):
        pass
    else:
        df1.to_csv(config["save_perturbed_data"]["path"] +
                   "/"+dataset_name+"sentence_jumbling.csv")

    embeddings = {"embed_original": embed(
        df1["original_sentence"], encoder_model, encoder_model_name)}
    for idx in range(n):
        embeddings[f"n={idx+1}"] = embed(df1[f"n={idx+1}"],
                                         encoder_model, encoder_model_name)
    similarity_score = {}
    for idx in range(n):
        similarity_score[f"original_sentence-n={idx+1}"] = similarity_between_sent(
            embeddings["embed_original"], embeddings[f"n={idx+1}"])

    end = time.time()
    exec_time = end-start

    writeout(model=encoder_model_name, task="jumbling", dataset=dataset_name,
             sim_score=list(similarity_score.values()), execution_time=exec_time)


if __name__ == "__main__":

    # dataset setup for testing paraphrasing hypothesis
    # calling the dataset
    mrpc_data = load_data(config["dataset_config"]["MRPC"]["path"])
    qqp_data = load_data(config["dataset_config"]["QQP"]["path"])
    paws_wiki_data = load_data(config["dataset_config"]["paws-wiki"]["path"])

    #sampling the data for further hypothesis testing.
    #mrpc data sampling for paraphrasing task
    mrpc_positive_sample = mrpc_data[mrpc_data.label == 1].sample(
        1194)  # sample label = 1 -> positive samples
    # sample label = 0 -> negative samples
    mrpc_negative_sample = mrpc_data[mrpc_data.label == 0]
    mrpc_sent1 = mrpc_positive_sample["sentence1"]
    mrpc_sent2 = mrpc_positive_sample["sentence2"]
    mrpc_sent3 = mrpc_negative_sample["sentence1"]
    mrpc_sent4 = mrpc_negative_sample["sentence2"]

    #paws-wiki data sampling for paraphrasing task
    paws_wiki_positive_sample = paws_wiki_data[paws_wiki_data.label == 1].sample(
        1194)
    paws_wiki_negative_sample = paws_wiki_data[paws_wiki_data.label == 0].sample(
        1194)
    paws_wiki_sent1 = paws_wiki_positive_sample["sentence1"]
    paws_wiki_sent2 = paws_wiki_positive_sample["sentence2"]
    paws_wiki_sent3 = paws_wiki_negative_sample["sentence1"]
    paws_wiki_sent4 = paws_wiki_negative_sample["sentence2"]

    #qqp data sampling for paraphrasing task
    qqp_positive_sample = qqp_data[qqp_data.is_duplicate == 1].sample(1194)
    qqp_negative_sample = qqp_data[qqp_data.is_duplicate == 0].sample(1194)
    qqp_sent1 = qqp_positive_sample["question1"]
    qqp_sent2 = qqp_positive_sample["question2"]
    qqp_sent3 = qqp_negative_sample["question1"]
    qqp_sent4 = qqp_negative_sample["question2"]

    # data sampling for rest of the  hypothesis testing
    mrpc_original_sent = mrpc_data.sentence1.sample(3500)
    qqp_original_sent = qqp_data.question1.sample(3500)
    paws_wiki_original_sent = paws_wiki_data.sentence1.sample(3500)

    # Loading models
    replace = WordReplacer()
    jumbler = WordSwapping()
    use = USE()
    laser = LASERMODEL(config["model_config"]["laser"]["path"])
    inferSent = infersent()
    
    all_models = [use, laser, inferSent]
    all_model_name = ["use", "laser", "InferSent"]

    for idx, models in enumerate(all_models):
        # MRPC DATASET
        paraphrasing_task(models, all_model_name[idx], "mrpc", mrpc_sent1,
                          mrpc_sent2, mrpc_sent3, mrpc_sent4)
        synonym_and_antonym_replacement(models, task="synonyms", encoder_model_name=all_model_name[idx],
                                        dataset_name="mrpc", original_sentence=mrpc_original_sent, n=3)
        synonym_and_antonym_replacement(models, task="antonyms", encoder_model_name=all_model_name[idx],
                                        dataset_name="mrpc", original_sentence=mrpc_original_sent, n=3)
        sentence_jumbling(models, encoder_model_name=all_model_name[idx],
                          dataset_name="mrpc", original_sentence=mrpc_original_sent, n=3)

        # QQP DATASET
        paraphrasing_task(models, all_model_name[idx], "qqp", qqp_sent1,
                          qqp_sent2, qqp_sent3, qqp_sent4)
        synonym_and_antonym_replacement(models, task="synonyms", encoder_model_name=all_model_name[idx],
                                        dataset_name="qqp", original_sentence=qqp_original_sent, n=3)
        synonym_and_antonym_replacement(models, task="antonyms", encoder_model_name=all_model_name[idx],
                                        dataset_name="qqp", original_sentence=qqp_original_sent, n=3)
        sentence_jumbling(models, encoder_model_name=all_model_name[idx],
                          dataset_name="qqp", original_sentence=qqp_original_sent, n=3)

        # paws-wiki datasets
        paraphrasing_task(models, all_model_name[idx], "paws-wiki", paws_wiki_sent1,
                          paws_wiki_sent2, paws_wiki_sent3, paws_wiki_sent4)
        synonym_and_antonym_replacement(models, task="synonyms", encoder_model_name=all_model_name[idx],
                                        dataset_name="paws_wiki", original_sentence=paws_wiki_original_sent, n=3)
        synonym_and_antonym_replacement(models, task="antonyms", encoder_model_name=all_model_name[idx],
                                        dataset_name="paws_wiki", original_sentence=paws_wiki_original_sent, n=3)
        sentence_jumbling(models, encoder_model_name=all_model_name[idx], dataset_name="paws-wiki",
                          original_sentence=paws_wiki_original_sent, n=3)

    #calling sbert models with it variants.
    sbert_paraphrasing = SentBert("paraphrasing")
    sbert_syn = SentBert("synonym")
    sbert_anto = SentBert("antonym")
    sbert_jumb = SentBert("jumbling")

    sbert_models = [sbert_paraphrasing, sbert_syn, sbert_anto, sbert_jumb]
    for model in sbert_models:
        # #paraphrasing task testing
        # MRPC dataset
        paraphrasing_task(model, "sbert", "mrpc", mrpc_sent1,
                          mrpc_sent2, mrpc_sent3, mrpc_sent4)
        synonym_and_antonym_replacement(model, task="synonyms", encoder_model_name="sbert",
                                        dataset_name="mrpc", original_sentence=mrpc_original_sent, n=3)
        synonym_and_antonym_replacement(model, task="antonyms", encoder_model_name="sbert",
                                        dataset_name="mrpc", original_sentence=mrpc_original_sent, n=3)
        sentence_jumbling(model, encoder_model_name="sbert",
                          dataset_name="mrpc", original_sentence=mrpc_original_sent, n=3)
        # qqp dataset
        paraphrasing_task(model, "sbert", "qqp", qqp_sent1,
                          qqp_sent2, qqp_sent3, qqp_sent4)
        synonym_and_antonym_replacement(model, task="synonyms", encoder_model_name="sbert",
                                        dataset_name="qqp", original_sentence=qqp_original_sent, n=3)
        synonym_and_antonym_replacement(model, task="antonyms", encoder_model_name="sbert",
                                        dataset_name="qqp", original_sentence=qqp_original_sent, n=3)
        sentence_jumbling(model, encoder_model_name="sbert",
                          dataset_name="qqp", original_sentence=qqp_original_sent, n=3)
        # paws-wiki dataset
        paraphrasing_task(model, "sbert", "paws-wiki", paws_wiki_sent1,
                          paws_wiki_sent2, paws_wiki_sent3, paws_wiki_sent4)
        synonym_and_antonym_replacement(model, task="synonyms", encoder_model_name="sbert",
                                        dataset_name="paws_wiki", original_sentence=paws_wiki_original_sent, n=3)
        synonym_and_antonym_replacement(model, task="antonyms", encoder_model_name="sbert",
                                        dataset_name="paws_wiki", original_sentence=paws_wiki_original_sent, n=3)
        sentence_jumbling(model, encoder_model_name="sbert", dataset_name="paws-wiki",
                          original_sentence=paws_wiki_original_sent, n=3)

    print("Done")
