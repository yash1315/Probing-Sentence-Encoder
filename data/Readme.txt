Dataset downloadable link.
Note that: the perturbed data is created with
the help these dataset and the code.

To download the dataset, please follow the link:

1) paws-wiki : https://github.com/google-research-datasets/paws
	Open the link and scroll to PAWS_Wiki section. Download
	PAWS-Wiki Labeled (Final)

2) QQP:  https://huggingface.co/datasets/glue/viewer/qqp/train
	Visit the huggingface link and download qqp paraphrasing 
	train dataset. 

3) MRPC: https://huggingface.co/datasets/glue/viewer/mrpc/train
	To download Microsoft Research Paraphrasing Corpus(MRPC) 
	dataset, visit the link and download the dataset (train 
	version).
Alternative: 
You can use the dataset provided in the zip file. Just unzip the data file
and use the data. 

Perturbed Data Generation:
We used the above dataset to create sentence perturbation for hypothesis 
testing. We took the first column (i.e. sentence1 or question1) as our original
sentence and produce a sentence perturbation for these sentences using 
WordNet toolkit. The code is provided in the zip file. 
check: scr/word_replacer.py

