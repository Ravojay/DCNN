import nltk
from nltk import word_tokenize
from itertools import chain, groupby
import numpy as np
import os
from collections import defaultdict

source_data = 'ag_news_csv'
dest_data = 'agnews'
dict_dir = 'glove.42B.300d.txt'

def clearText():
    print 'Cleaning Data...'
    data = []
    label_fid = open(dest_data + '/train_label.txt', 'w')
    #remove  useless symbols
    count = 0
    lines = []
    with open(source_data + '/train.csv') as f:
        for line in f:
            line = line.strip('\n')
            line = line.split('\",\"')
            line[1] = line[1].replace('\\',' ').replace('\"',' ').replace('/',' ').replace('\'',' ').replace('*',' ').replace('-',' ').replace('.',' ').replace('(', ' ').replace(')', ' ')
            line[2] = line[2].replace('\\',' ').replace('\"',' ').replace('/',' ').replace('\'',' ').replace('*',' ').replace('-',' ').replace('.',' ').replace('(', ' ').replace(')', ' ')
            # subtract the offset of labels
            lines.append(str(int(line[0].split('\"')[1])-1)+'\n')#label 1-5 to 0-4
            data.append(line[1]+' '+line[2])
            count += 1
    lines.insert(0,str(count)+"\n")
    s=''.join(lines)
    label_fid.write(s)
    train_num = len(data)
    label_fid.close()

    label_fid = open(dest_data+'/test_label.txt', 'w')
    count = 0
    lines = []
    with open(source_data+'/test.csv') as f:
        for line in f:
            line = line.strip('\n')
            line = str(line)
            line = line.split('\",\"')
            line[1] = line[1].replace('\\',' ').replace('\"\"',' ').replace('/',' ').replace('\'',' ').replace('*',' ').replace('-',' ').replace('.',' ').replace('(', ' ').replace(')', ' ')
            line[2] = line[2].replace('\\',' ').replace('\"\"',' ').replace('/',' ').replace('\'',' ').replace('*',' ').replace('-',' ').replace('.',' ').replace('(', ' ').replace(')', ' ')
            # subtract the offset of labels
            lines.append(str(int(line[0].split('\"')[1])-1)+'\n')
            data.append(line[1]+' '+line[2].rstrip('\"'))
            count += 1

    lines.insert(0,str(count)+"\n")
    s=''.join(lines)
    label_fid.write(s)
    label_fid.close()

    return data, train_num

def gen_global_dict():
    print 'Generating global dict... It may take a few of seconds.'
    global_dict = {}
    with open(dict_dir) as f:
        for line in f:
            line = line.strip('\n')
            line = line.split(' ', 1)
            global_dict[line[0]] = line[1]
    return global_dict

def gen_local_dict(global_dict, data):
    print 'Generating local dict...'
    separate_str = ' '
    tmp = separate_str.join(data)
    wordtoken = nltk.tokenize.word_tokenize(tmp.lower())
    #select the top 50k begin
    wordsNb = 100000#here to set the number of words
    myDict = defaultdict(int)
    for word in wordtoken:
        myDict[word] += 1
    words = list(map(lambda x : x[0], sorted(myDict.items(), key=lambda item : item[1], reverse = True)[:wordsNb]))
    #select the top 10k end
    #words = list(np.unique(wordtoken))
    print str(len(words))+' different words in total.'

    local_dict = {}
    local_dict_fid = open(dest_data+'/local_dict.txt', 'w')
    local_fail_fid = open(dest_data+'/fail.txt', 'w')
    index = 1
    local_dict_fid.write(' '.join(['0.0' for _ in range(300)])+'\n')#index=0:padding's feature 300 zeros
    for i in range(0, len(words)):
        if global_dict.has_key(words[i]):
            local_dict[words[i]] = index
            index = index + 1
            local_dict_fid.write(global_dict[words[i]]+'\n')
        else:
            local_fail_fid.write(words[i]+'\n')
    print 'Local dict has been saved in \'local_dict.txt\'.'
    print 'Words that cannot be found in dict are saved in the \'fail.txt\''
    local_dict_fid.close()
    local_fail_fid.close()
    return local_dict

def gen_train_test(local_dict, data, train_num, max_words_length=128):
    train_fid = open(dest_data + '/train_data.txt', 'w')
    test_fid = open(dest_data + '/test_data.txt', 'w')
    print 'Begin to generate train/test data.'
    for i in range(len(data)):
        if i % 10000 == 0:
            print str(i) + ' samples have been generated!'
        words = nltk.tokenize.word_tokenize(data[i].lower())
        words_length_limit = max_words_length
        j = 0
        if i < train_num:
            while j < len(words) and j < words_length_limit:
                if local_dict.has_key(words[j]):
                    train_fid.write(str(local_dict[words[j]])+' ')
                else:
                    words_length_limit += 1
                j += 1
            while j < words_length_limit:
                train_fid.write('0 ')
                j += 1
            train_fid.write('\n')
        else:
            while j < len(words) and j < words_length_limit:
                if local_dict.has_key(words[j]):
                    test_fid.write(str(local_dict[words[j]])+' ')
                else:
                    words_length_limit += 1
                j += 1
            while j < words_length_limit:
                test_fid.write('0 ')
                j += 1
            test_fid.write('\n')
    print str(len(data)) + ' samples have been generated!'
    train_fid.close()
    test_fid.close()

if __name__ == '__main__':
    '''
    step1: convert pretrained word vectors to global_dict - gen_global_dict().
    step2: clear the training and testing data - clearText().
    step3: select the words which will be used for training and testing, and generate local dict = gen_local_dict(global_dict, data)
    step4: generate training and testing data in an index format - gen_train_test(local_dict, data, train_num)
    The example of results can be found in $dest_data
    '''
    global_dict = gen_global_dict()
    if (not os.path.exists(dest_data)):
        os.system('mkdir '+dest_data)
    data, train_num = clearText()
    print 'There are '+str(train_num)+' training samples and '+str(len(data) - train_num)+' testing samples.'
    local_dict = gen_local_dict(global_dict, data)
    gen_train_test(local_dict, data, train_num)
    print 'Finished!'




