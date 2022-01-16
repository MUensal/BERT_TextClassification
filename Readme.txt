Read-Me for the Code
A short information on how to use run the Modell is given here. 

BERT Modell

Preprocessing

The Processing Pipeline starts with the File 

‘Training_data’, which is processed in the 
‘Data_Preparation.py’ class. The output of this class is called 
‘Prepared_data’ and is processed in 
‘Preprocessing_Pipeline.py’. The output here is called 
‘Pre_processed_data’. In this file, I showcase some preprocessing methods and I also save the stopwords in a file called 
‘nltk_german_stop_words.txt', however, this one is not processed any further. 
‘Bert_preprocess.py’ includes some experimental pre-processing before starting model implementation and can even be ignored in this pipeline. 


Modell Implementation

The BERT Modell is implemented in 
‘BERT_training.py’ with the 
‘Prepared_data’ file, 

where it finally does output the classification results. 

__________________________________________

DEPENDENCIES

- transformers 
- sklearn
- torch
- pandas
- tqdm
- matplotlib 
- seaborn

keras                   2.6.0
matplotlib              3.4.2
nltk                    3.6.2
numpy                   1.19.5
pandas                  1.2.4
pip                     21.1.1
regex                   2021.4.4
sacremoses              0.0.45
scikit-learn            0.24.2
seaborn                 0.11.1
sklearn                 0.0
tokenizers              0.10.2
torch                   1.8.0
tqdm                    4.60.0
transformers            4.6.0


