import json
from gensim.models import Word2Vec
# data = json.load(open('results_val.json'))
# sentences = []
# #len(data["results"]) : 40504
# for i in range(len(data["results"])):
#     for sentence in data["results"][i]["captions"]:
#         sentences += [sentence.split(' ')]
# #print(sentences)
# model = Word2Vec(sentences, min_count=1)
# model.save('trainedModel')
new_model = Word2Vec.load('trainedModel')
say_vector = new_model['red']  # get vector for word
print(new_model.wv.similarity('blue', 'sky'))
print(say_vector)



# from gensim.models import Doc2Vec

# sentences = ['hi this is ravi','hello hi'],

# model = Doc2Vec(sentences)

# # store the model to mmap-able files
# model.save('/tmp/my_model.doc2vec')
# # load the model back
# model_loaded = Doc2Vec.load('/tmp/my_model.doc2vec')

# import gensim, logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
 
# sentences = [['first', 'sentence'], ['second', 'sentence']]
# # train word2vec on the two sentences
# model = gensim.models.Word2Vec(sentences, min_count=1)
# print(model['first'])