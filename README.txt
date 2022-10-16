To execute the code correct way, please follow the instructions mentioned 
below:
1) Intall the requirements.txt file. To install the file, type in command line
	" python3 -m pip install -r requirments.txt"
this command will help you install all the dependies required to run the code
successfully. 
2)Unzip the code file and the data file. Note that, I have also provided the 
same data file with the code zip file, just to aviod any malfunction. 
Now, simply execute the main.py file and veiw the results. 

Models installation: 
to run the code smoothly, please save the models in the instructed locations: 
1) InferSent Model:

1.1) Glove embeddings:
follow the instruction 
cd models/GloVe
!curl -Lo GloVe/glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip
!unzip GloVe/glove.840B.300d.zip -d GloVe/

1.2) encoder
copy and paste the command on command line and save it at models/InferSent/encoder 
!curl -Lo encoder/infersent1.pkl https://dl.fbaipublicfiles.com/infersent/infersent1.pkl
!curl -Lo encoder/infersent2.pkl https://dl.fbaipublicfiles.com/infersent/infersent2.pkl

3) Note that: you may not be able to see Doc2vec results from main.py files,
however, to see the execute and the results of Doc2Vec, please use the ipyton
notebook attach with the submission. 

Alternative option:

Unzip the ipython notebook attached with the submission and follow the read me
file inthere. 