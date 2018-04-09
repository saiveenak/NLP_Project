import json
from gensim.models import Word2Vec

def get3Dmatrix():
    model = Word2Vec.load('trainedModel')
    data = json.load(open('test.json'))
    img_name = "COCO_val2014_000000303994.jpg"

    columns = len(model['a'])

    top10captions, maxCaptionLength= [], 0
    for sentence in data["results"][0]["captions"][:10]:
        maxCaptionLength = max(maxCaptionLength, len(sentence.split(' ')))
        top10captions += [sentence.split(' ')]

    vectors = []
    once = True
    # print(top10captions)
    # print(maxCaptionLength)
    for sentence in top10captions:
        sentenceVector = []
        for token in sentence:
            wordVector = model[token]
            sentenceVector.append(list(wordVector))
        for i in range(maxCaptionLength - len(sentence)):
            wordVector = [0] * columns
            sentenceVector.append(wordVector)
        vectors.append(sentenceVector)

    # with open('sample.txt','w') as f:
    #     f.write(str(vectors))
    # print((vectors[2][3]) == list(model['red']))
    # print(len(vectors[4][6]))
    return vectors, maxCaptionLength

