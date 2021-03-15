
import nltk
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize
import numpy
import tflearn
import tensorflow



ps = nltk.stem.PorterStemmer()
# read my data set
dict = {}
index = 0
toRead = ["GAME_RULES.txt", "SHOPPING.txt", "STORE_DETAILS.txt", "WHAT_IS_IT.txt"]
# toRead = ["GAME_RULES.txt"]
for fileName in toRead:

    f = open("dialogues/" + fileName, "r")


    text = f.readlines()

    for i in text:
        temp = eval(i)
        temp2 = {index: temp}
        dict.update(temp2)
        index += 1
    f.close()

print("---------------------------------------------" + str(len(dict)))
"""
final_dict = {1: dict{id:...., turns:[chat_dialogue]}
    ,1: dict{id:...., turns:[chat_dialogue],........498: dict{id:...., turns:[chat_dialogue]}


print(dict[0]["turns"])
# OUTPUT
first conversation in dataset 
"""


# print("+++++++++++++++")
# print(dict[0]["turns"])
# print(len(dict[0]["turns"]))
# print("+++++++++++++++")

dual_set = []

for i in range(len(dict)):
    #skip bots greeding
    # print("\n\n\nread file\n")
    # use "len(dict[i]["turns"])-1" because some times the user answer but bot didn't answer back (posibility for silent answer)
    for j in range(1, len(dict[i]["turns"])-1, 2):
        dual_set.append([dict[i]["turns"][j], dict[i]["turns"][j+1]])
        # print("j = ", j)
        # print("input ", [dict[i]["turns"][j], dict[i]["turns"][j+1]])
    # print("\n\n\nend reading ")



dual_set_copy = dual_set.copy()
# k = sorted(dual_set, key=lambda x: x[1])
# k = dual_set

# print("-----------")
# for i in k:
#     print(i)
# print(len(k))
# print("------------")

"""
dual set = [[Q1, Awn1], [Q2,Awn2], ..... [Q3,Awn3] ]
in this set we haven't saved bots initial greeting

"""


stopwords_english = stopwords.words('english')



# Create bag with all unique words in it
all_words = []

# because tokenize alters the original list and I might don't want that
for i in range(len(dual_set_copy)):
    # always will be 2
    for k in range(len(dual_set_copy[i])):
        words = word_tokenize(dual_set_copy[i][0])
        for j in words:
            word = j.lower()
            # users put "/" instead of "?"
            if word[-1] == "/":
                word = word[:-1]
            if word not in all_words and word not in string.punctuation:
                all_words.append(word)




"""
for removing stopwords and punctuation from the sentences

Input string and list of stopwords

OUTPUT: [sentence, separated, with, stem, and, no, stopwords]
yu can't help me with a kids game? -> ['yu', 'ca', "n't", 'help', 'kid', 'game']
"""
def cleanSentence(sen_or, my_stopwords_english = stopwords_english):
    sen_clean = []
    sen = word_tokenize(sen_or)
    for word in sen:
        word = word.lower()
        if word not in my_stopwords_english and word not in string.punctuation:
            word = ps.stem(word)
            sen_clean.append(word)
    return sen_clean
# print(cleanSentence("yu can't help me with a kids game?"))


"""
input stem sen

INPUT: ['play', 'monopoli'] 
OUTPUT: "play monopoli "


INPUT:  [] 
OUTPUT: ""
"""
def gen_class(list_of_words):
    c = ""
    for i in list_of_words:
        c = c + str(i)
        c = c + " "
    # print(c)
    return c




# dict with "class" indedification and awnswers
q_and_a_dict = {}

"""
for class id it will be the user's sentence stem + no stopwords + no punctuation <each word separated by space>

may also add same answers but will see about that
"""
for i in range(len(dual_set)):
    stem_sen = cleanSentence(dual_set[i][0])
    my_class = gen_class(stem_sen)

    # if new entry in dict insert else update old one's answers
    if my_class in q_and_a_dict:
        temp_list = q_and_a_dict[my_class]
        temp_list.append(dual_set[i][1])
        q_and_a_dict.update({my_class: temp_list})
    else:
        q_and_a_dict.update({my_class: [dual_set[i][0], dual_set[i][1]]})


# import collections
# od = collections.OrderedDict(sorted(q_and_a_dict.items()))
# for k, v in od.items():
#     print(k, "   ", v)
#
#


# for i in q_and_a_dict.keys():
#     if len(q_and_a_dict[i]) > 0:
#         print(i, " : ", q_and_a_dict[i])

# for i in range(len(dual_set)):
#     print(dual_set[i])
#     print()

y_labels = []
x_input = []
labels = [0 for _ in range(len(q_and_a_dict))]

for index, user_input_class in enumerate(q_and_a_dict):
    bag = []
    words_of_user_input = word_tokenize(q_and_a_dict[user_input_class][0])

    for i in all_words:
        if i in words_of_user_input:
            bag.append(1)
        else:
            bag.append(0)

    temp_labels = labels[:]
    temp_labels[index] = 1

    y_labels.append(temp_labels)
    x_input.append(bag)


y_labels = numpy.array(y_labels)
x_input = numpy.array(x_input)


def bag_of_words(s, all_the_words = all_words, debug = False):
    bag = [0 for _ in range(len(all_the_words))]

    # s_words = nltk.word_tokenize(s)
    # s_words = [stemmer.stem(word.lower()) for word in s_words]
    # s_words = cleanSentence(s)

    token_sen = word_tokenize(s)
    final_input = []

    for j in token_sen:
        word = j.lower()
        # users put "/" instead of "?"
        if word[-1] == "/":
            word = word[:-1]
        final_input.append(word)

    if debug:
        print("trimmed sentence = ", final_input)

    for se in final_input:
        for i, w in enumerate(all_the_words):
            if w == se:
                bag[i] = 1
    return numpy.array(bag)


#
# print("\n\n\n\n DEBUGGGGGGGGGG \n\n")
#
# for index,i in enumerate(all_words):
#     print(index , " : ", i)
#
#
# print(y_labels[-1])
# print(x_input[-1])
#
# for index, i in enumerate(x_input[-1]):
#     if i == 1:
#         print(index, i, end= " ")
#         print(all_words[index])
#
# print(dual_set[-1])
#
#
# h = bag_of_words("Okay. Can you tell me how to win the game?")
# print("h= ",h)

# for index, i in enumerate(h[-1]):
#     if i == 1:
#         print(index, i)
#         print(all_words[index])


# input()










import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.python.framework import ops

ops.reset_default_graph()

net = tflearn.input_data(shape=[None, len(x_input[0])])
net = tflearn.fully_connected(net, len(x_input[0]))
net = tflearn.fully_connected(net, len(x_input[0]))
net = tflearn.fully_connected(net, len(x_input[0]))
net = tflearn.fully_connected(net, len(y_labels[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)





from pathlib import Path

if Path('chatBot2.tflearn.meta').is_file():
    print("loading Model")
    model.load("chatBot2.tflearn")
else:
    # print("File not exist")
    model.fit(x_input, y_labels, n_epoch=20, batch_size=1000, show_metric=True)
    model.save("chatBot2.tflearn")


# try:
#     model.load("chatBot.tflearn")
# except:
#     model.fit(x_input, y_labels, n_epoch=100, batch_size=1000, show_metric=True)
#     model.save("chatBot.tflearn")




import random





print("\n\n\n")

while True:
    x = input("\nwaiting for input:     ")
    if x.lower() == "quit":
        break

    senTon = bag_of_words(x, all_words, False)
    res = model.predict([senTon])
    # print(res)
    results_index = numpy.argmax(res)
    # print(results_index)

    keys = []
    for i in q_and_a_dict.keys():
        keys.append(i)

    k = random.randint(1, len(q_and_a_dict[keys[results_index]])-1)
    # print(keys[results_index], ": ", q_and_a_dict[keys[results_index]])
    print(q_and_a_dict[keys[results_index]][k])

