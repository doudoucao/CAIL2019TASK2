# fin = open('./data/small_data/divorce/data_small_selected.json', 'r', encoding='utf-8')
import json
import http.client
import hashlib
import json
import urllib
import random

def init(tags_path):
    f = open(tags_path, 'r', encoding='utf8')
    tag_dic = {}
    tagname_dic = {}
    line = f.readline()
    while line:
        tagname_dic[len(tag_dic)] = line.strip()
        tag_dic[line.strip()] = len(tag_dic)
        line = f.readline()
    f.close()
    return tag_dic, tagname_dic


def getlabel(sent, tag_dic):
    if len(sent['labels']) > 0:
        labels = [0] * len(tag_dic)
        for i in sent['labels']:
            labels[tag_dic[i]] = 1
        return labels
    return [0]*len(tag_dic)


def read_trainData(path, tag_path, output_path=None):
    fin = open(path, 'r', encoding='utf8')
    fout = open(output_path, 'w', encoding='utf-8')
    tag_dic, tagname_dic = init(tag_path)

    all_text = []
    tag_label = []
    line = fin.readline()
    i = 1
    while line:
        d = json.loads(line)
        for sent in d:
            all_text.append(sent['sentence'])
            labels = getlabel(sent, tag_dic)

            # if labels == [0]*len(tag_dic):
            #     continue

            # back translation
            '''
            tmp = baidu_translate(str(sent['sentence']).strip().replace('\n', '').replace('\t', ''), fromLang='zh', toLang='en')
            sentence = baidu_translate(str(tmp), fromLang='en', toLang='zh')

            if labels[15] == 1:
                fout.write(
                    str(i) + '\t' + str(sent['sentence']).strip().replace('\n', '') + '\t' + ' '.join([str(i) for i in labels]) + '\n')
            '''
            fout.write(str(i) + '\t' + str(sent['sentence']).strip().replace('\n', '').replace('\t', '') + '\t' + ' '.join([str(i) for i in getlabel(sent, tag_dic)]) + '\n')
        line = fin.readline()
        i += 1
    fin.close()

# read_trainData('./data/data_big/labor/train_selected.json', './data/data_big/labor/tags.txt', './data/data_big/labor/labor_trans.txt')
# read_trainData('./data/data_big/loan/train_selected.json', './data/data_big/loan/tags.txt', './data/data_big/loan/loan_train.txt')
# read_trainData('./data/data_big/divorce/train_selected.json', './data/data_big/divorce/tags.txt', './data/data_big/divorce/divorce_train.txt')

# combine dataset
def combine_data(in_path, out_path):
    fin = open(in_path, 'r', encoding='utf8')
    fout = open(out_path, 'w', encoding='utf-8')
    line = fin.readline()
    while line:
        content = line.split('\t')
        id = content[0]
        text = content[1]
        labels = [int(i) for i in content[2].split()]
        fout.write(str(id) + '\t' + str(text) + '\t' + ' '.join([str(i) for i in labels]) + '\n')
        line2 = fin.readline()
        if line2 != '':
            # print(line2)
            content2 = line2.split('\t')
            # print(content2)
            id2 = content2[0]
            text2 = content2[1]
            labels2 = [int(i) for i in content2[2].split()]
            fout.write(str(id2) + '\t' + str(text2) + '\t' + ' '.join([str(i) for i in labels2]) + '\n')
            if id == id2 and labels != [0]*20 and labels2 != [0]*20:
                for i, label in enumerate(labels2):
                    if label == 1:
                        labels[i] = 1
                fout.write(str(id2) + '\t' + str(text) + ',' + str(text2) + '\t' + ' '.join([str(i) for i in labels]) + '\n')
        line = fin.readline()
    fout.close()


# combine_data('./data/data_small/divorce/divorce_train.txt', './data/data_small/divorce/divorce_combine.txt')
#combine_data('./data/data_small/labor/labor_train.txt', './data/data_small/labor/labor_combine.txt')
# combine_data('./data/data_small/loan/loan_train.txt', './data/data_small/loan/loan_combine.txt')



'''
import pandas as pd

df = pd.read_csv('./data/divorce_train.txt', sep='\t', header=None)
print(df.values)
'''
# print(open('./data/data_small/divorce/tags.txt').readlines())
# baidu_translate('On June 1, 2016, the plaintiff told the court that he wanted to divorce, and the defendant agreed to divorce. He asked the plaintiff to raise Xue Mouping, a married son, and pay the maintenance fee.')
