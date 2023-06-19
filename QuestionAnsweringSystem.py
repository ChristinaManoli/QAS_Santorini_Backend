# Import dependencies
## for data
import json
import pandas as pd
import numpy as np
from sklearn import metrics, manifold
from sklearn.metrics.pairwise import cosine_similarity
## for processing
import re
import nltk
## for w2v

## for bert
from sentence_transformers import SentenceTransformer

import spacy
import truecase
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


lst_stopwords = nltk.corpus.stopwords.words("english")

#Preprocessing and Cleaning method 1
def utils_preprocess_text1(sentence):
  #Remove any < > references
  new_sentence = re.sub("<+[\w]*>", "", sentence)
  #Remove the word Santorini
  new_sentence = new_sentence.replace("Santorini", "")
  new_sentence = new_sentence.replace("santorini", "")

  return new_sentence

#Preprocessing and Cleaning method 2
def utils_preprocess_text2(text, flg_lemm=True, lst_stopwords=None):
  #Clean (convert to lowercase and remove punctuations and characters and then strip)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    #Tokenize (convert from string to list)
    lst_text = text.split()
    #Remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in
                    lst_stopwords]
    #Lemmatisation (convert the word into root word)
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]

    #Back to string from list
    text = " ".join(lst_text)
    return text

def Similarity_LinguisticApproach(question, templates):
  score = []
  for template in templates["Template_Clean3"]:
    sc=0
    for word in question.split():
      if word in template.split():
        sc=sc+1
    score.append(sc)
  return score

def Template_Matching(Question):
  #Import templates
  templates = pd.read_csv("Templates.csv", sep='|', engine='python')
  #Cleaning Templates with utils_preprocess_text1 + utils_preprocess_text2 AND Cleaning Question with utils_preprocess_text2
  index=0
  for sentence in templates["Template"]:
    sentence = utils_preprocess_text1(sentence)
    templates.at[index,'Template_Clean3'] = utils_preprocess_text2(sentence, flg_lemm=True)
    index=index+1

  Question_simil_clean= utils_preprocess_text2(Question, flg_lemm=True)

  #~~~~~~~~~~~~~~ Sentence Similarity using SentenceTransformers ~~~~~~~~~~~~~~

  #Importing the model
  model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
  template_embeddings = model.encode(templates["Template_Clean3"])
  question_embeddings = model.encode(Question_simil_clean)

  #For every question we calculate how similar it is to each template
  similarities2 = cosine_similarity(question_embeddings.reshape(1, -1), template_embeddings)

  for i in range(len(similarities2)):
    #Rescale so they sum = 1
    similarities2[i] = similarities2[i] / sum(similarities2[i])

  #Classify the label with highest similarity score
  predicted_prob = similarities2
  predicted = []
  second =[]

  for pred in predicted_prob:
    x = np.argmax(pred)
    predicted.append("T" + str(x))
    #getting the second biggest similarity
    temp = np.copy(pred)
    temp[x] = -1
    x2=np.argmax(temp)
    second.append("T" + str(x2))


  #~~~~~~~~~~~~~~ Linguistic Similarity Method ~~~~~~~~~~~~~~

  #Cleaning Templates with utils_preprocess_text1 + utils_preprocess_text2 AND Cleaning Question with utils_preprocess_text2
  index=0
  for sentence in templates["Template"]:
    sentence = utils_preprocess_text1(sentence)
    templates.at[index,'Template_Clean3'] = utils_preprocess_text2(sentence, flg_lemm=True, lst_stopwords=lst_stopwords)
    index=index+1

  Question_ling_clean = utils_preprocess_text2(Question, flg_lemm=True, lst_stopwords=lst_stopwords)

  scores = []
  scores = Similarity_LinguisticApproach(Question_ling_clean, templates)
  count=0
  for i in scores:
    count=count+i
  if count==0:
    scores.clear()

  predicted_prob = scores
  predicted2=[]
  if predicted_prob:
    x = np.argmax(pred)
    predicted2.append("T" + str(x))
  else:
    predicted2.append(None)


  Final_predictions=[]
  i=0
  for p1, p2 in zip(predicted, predicted2):
    if p2:
      if p2==p1:
        final_Pred = p1
      elif p2==second[i]:
        final_Pred = second[i]
      else:
        final_Pred = p2
    else:
      final_Pred = p1

    i=i+1
    Final_predictions.append(final_Pred)

  return Final_predictions, templates, second

#------------------ Answer Retrieval ------------------

def cypher_Query_Retrieval(Final_predictions, Question, templates):
  for p in Final_predictions:
    #fetching the appropriate template
    template_index = int(p[1])
    cypher_query = templates["Cypher_Clean"].iloc[template_index]
    if cypher_query.count("#") > 0:
      cypher_query = QA_NER(Question, cypher_query)
  return cypher_query

def QA_NER(Question, cypher_query):
  NER = spacy.load("en_core_web_sm")
  #Truecasing the questions in order to apply NER method 
  n = truecase.get_true_case(Question)
  #Using NER to find the key entity that is required in the cypher query
  entities = NER(n)
  if cypher_query.count("#") == 1:
    for word in entities.ents:
      cypher_query = cypher_query.replace("#", str(word))
  elif cypher_query.count("#") == entities.ents.count:
    indexes = [x.start() for x in re.finditer('\#', cypher_query)] 
    for i, word in zip(indexes, entities.ents):
      cypher_query = cypher_query[:i] + str(word) + cypher_query[i+1:]      
  return cypher_query


def recommendation_retrieval(Final_predictions, previous_question, templates):
  for p in Final_predictions:
    #fetching the appropriate template
    template_index = int(p[1])
    cypher_query = templates["Cypher_Clean"].iloc[template_index]

    NER = spacy.load("en_core_web_sm")
    n = truecase.get_true_case(previous_question)
    #Using NER to find the key entity that is required in the cypher query
    entities = NER(n)

    if cypher_query.count("#") > 0:
      for word in entities.ents:
        cypher_query = cypher_query.replace("#", str(word))
      
  return cypher_query


def Answer_Retrieval(Question, templates, Final_predictions, second):
  index=0
  for cyph_q in templates["Cypher"]:
    new_cyph_q = re.sub("<+[\w]*>", "#", cyph_q)
    templates.at[index,'Cypher_Clean'] = new_cyph_q
    index=index+1
  
  answer = cypher_Query_Retrieval(Final_predictions, Question, templates)

  if Final_predictions != second:
    #fetching the appropriate template
    template_index = int(second[0][1])
    nl = templates["Template"].iloc[template_index]
    
    answer2 = recommendation_retrieval(second, Question, templates)
      
  return answer, nl, answer2




@app.route('/api/QASantorini', methods=['POST'])
def qa_endpoint():
    Question = request.json['argument']
    
    # Call your machine learning script with the argument
    Final_Pred, Templates, Second = Template_Matching(Question)
    cypherQuery, FollowUpRecQuestion, FollowUpRec_CypherQuery = Answer_Retrieval(Question, Templates, Final_Pred, Second)
    
    # Return the results as a JSON response
    response = {
        'result1': cypherQuery,
        'result2': FollowUpRecQuestion,
        'result3': FollowUpRec_CypherQuery
    }
    return jsonify(response)


#Run the Flask app
if __name__ == '__main__':
  app.run()