from googletrans import Translator
from tqdm import tqdm
import os
import random
import time
import re
import json


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

            if labels == [0]*len(tag_dic):
                continue

            # fout.write(str(i) + '\t' + str(sent['sentence']).strip().replace('\n', '').replace('\t', '') + '\t' + ' '.join([str(i) for i in getlabel(sent, tag_dic)]) + '\n')
            fout.write(str(i) + '\t' + str(sent['sentence']).strip().replace('\n', '').replace('\t', '') +  '\t' + ' '.join([str(i) for i in getlabel(sent, tag_dic)])  + '\n')
            # fout.write(str(sent['sentence']).strip().replace('\n', '').replace('\t', '') + '\n')
        line = fin.readline()
        i += 1
    fin.close()

# read_trainData('./data/data_big/loan/train_selected.json', './data/data_big/loan/tags.txt', './data/data_big/loan/loan_trans.txt')


import docx


def read_docx():
    f = open('./data/data_big/loan/loan_backtrans.txt', 'w')
    data = docx.Document('./data/data_big/loan/loan_en.docx')
    for index, para in enumerate(data.paragraphs):
        f.write(para.text +'\n')

# read_docx()

def read_backtrans():
    f = open('./data/data_big/loan/loan_trans.txt', 'r')
    f1 = open('./data/data_big/loan/loan_backtrans.txt', 'r')
    f2 = open('./data/data_big/loan/loan_trans_labels.txt', 'w')
    f1_lines = f1.readlines()
    i = 0
    for line in f:
        idx = line.split('\t')[0].strip()
        labels = line.split('\t')[2].strip()
        f2.write(idx + '\t' + f1_lines[i].strip() + '\t' + labels.strip() + '\n')
        i += 1

read_backtrans()