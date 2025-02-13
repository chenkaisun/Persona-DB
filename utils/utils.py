import json
import os
import time
import argparse
import os
import pickle as pkl
import json
# import time
import re
# import os
import numpy as np
# from bs4 import BeautifulSoup
# import argparse
# import gc
# from copy import deepcopy
# import torch.nn as nn
# import torch.nn.functional as F
# from pynvml import *
# import requests
# from numba import jit
from matplotlib.pyplot import plot
import logging
# from pynvml import *
import codecs
from IPython import embed
from glob import glob
logging.getLogger('matplotlib.font_manager').disabled = True
from skmultilearn.model_selection import iterative_train_test_split
import matplotlib.pyplot as plt
import requests
from collections import Counter
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
# import  wandb
import math
from sklearn.model_selection import train_test_split

# from gensim.parsing.preprocessing import remove_stopwords, STOPWORDS
# import torch
import pandas as pd
# import preprocessor as p

def parse_gpt_response(response,pure_int=False):
    sent_map = {"positive": 2, "neutral": 1, "negative": 0}
    sent_map_reverse = {0: "Negative", 1: "Neutral", 2: "Positive"}
    """an example response is a string 'Comment:hello world, Sentiment Polarity:1, Sentiment Intensity:1'
    we want to extract these three fields"""
    tmp_response=response
    response = response.strip()
    response = response.split("\n")
    response = [item for item in response if item.strip()]
    # print(response)
    response = [item.strip() for item in response]
    if len(response)>3 and "Comment" in response[-3]:
        response = response[-3:]

    # print(response)
    response = [item.split(":") for item in response]
    # print(response)
    # response = {item[0]: item[1] for item in response}r
    try:
        res = [item[1].strip() for item in response]
        res[1] = res[1].split()[0].strip()  # sentiemtn: 2 (like that..)
        res[2] = res[2].split()[0].strip()  # sentiemtn: 2 (like that..)
    except Exception as e:
        print(e)
        print(tmp_response)
        res = [".", "Neutral", 1]
        # exit()

    if res[1].lower() not in sent_map:
        print("here")
    res[1] = sent_map[res[1].lower()] if res[1].lower() in sent_map else 0

    try:
        res[2] = int(res[2])
    except:
        print("assigning res[2] = int(res[2])")
        # embed()
        print(res)
        res[2] = res[2][0]

    try:
        res[2] = int(res[2])
    except:
        print("The second res[2] = int(res[2])")
        res[2]=1

    if not pure_int:
        res[2] = int(res[2]) * (res[1] - 1) + 3
        if res[2] < 0 or res[2] > 6:
            print("here")
            res[2] = 4
    else:
        print("pure int")
        if res[2] < 0 or res[2] > 3:
            print("here")
            res[2] = 1

    return res

def contains_phrase(text, phrases):
    for phrase in phrases:
        if phrase in text:
            return True
    return False
def get_prompt(history=None, post=None, profile=None, key=None, nbs=None, sc=None, prompt_type=None, refine=None,neighbors=None,choices=None, prompts_dir="prompts/"):  # , prompt_template=None
    prompt_template = load_file(f"{prompts_dir}{prompt_type}.txt")
    # print(";paded")
    prompt=""
    if prompt_type == "pooling":
        prompt = prompt_template.format(nb1=nbs)
    elif prompt_type in ["extraction", "extraction_refine"]:
        prompt = prompt_template.format(history=history, profile=profile)
    elif prompt_type == "refine":
        prompt = prompt_template.format(history=history, profile=profile, key=key)
    elif prompt_type == "clean_anno":
        prompt = prompt_template.format(input=key)
    else:
        prompt = prompt_template.format(history=history, post=post, profile=profile, key=key, refine=refine,choices=choices,  neighbors=neighbors)
    return prompt






def parse_gpt_anno(response, to_dict=False):
    #get rid of None's

    tmp = response.strip().split("\n")
    tmp = [item.strip() for item in tmp if item.strip()]

    dic = {}
    try:
        res=[]
        for item in tmp:
            category, value=item.split(":", 1)
            tmp_category = category.strip().lower()
            value = value.strip()
            if not value:
                continue

            if ("issue" in tmp_category or "entit" in tmp_category):
                tmp_res = []
                stance_values = value.split(";")
                for stance_value in stance_values:
                    stance_value = stance_value.strip()
                    if "none" not in stance_value.lower():
                        tmp_res.append(stance_value)
                        if tmp_category not in dic:
                            dic[tmp_category] = {}
                        stance,entliststr=stance_value.split(":")
                        dic[tmp_category][stance.lower().strip()]=entliststr
                if not tmp_res:
                    continue
                tmp_res = " ; ".join(tmp_res)
                res.append(f"{category.strip()}: {tmp_res}")
            elif "none" not in value.lower(): #("issue" in category.lower() or "entit" in category.lower()) or
                res.append(item)

                if tmp_category in ["human value", "moral value"]:
                    tmp_category+="s"
                if value[-1]==";":
                    value=value[:-1]
                dic[tmp_category]=value



        if to_dict: return dic
        return "\n".join(res)
    except Exception as e:
        # print(e)
        if to_dict: return dic
        return response

def preprocess_user_tweet(text=None):
    new_text = []
    is_username=False
    for t in text.split(" "):
        if is_username:
            is_username=False
            continue
        if t.strip().lower() in ["@cnnbrk", "@cdcgov", "@bbcbreaking"]:
            continue
        t = '@user' if t.strip().startswith('@') else t
        t = '<url>' if 'http' in t.strip() else t #.startswith('http')        new_text.append(t)

        if t.strip().lower().startswith('ig:') or t.strip().lower().startswith('instagram'):
            is_username=True
            continue

        # if "\n" in t:
        #     tmp_list=[]
        #     tmp_tks=t.split("\n")
        #     for tmp_tk in tmp_tks:
        #         tmp_list.append(tmp_tk.strip())
        #     t=" ".join(tmp_list)


        new_text.append(t)

    # p.set_options(p.OPT.URL, p.OPT.MENTION,p.OPT.NUMBER)
    res= " ".join(new_text)

    validate_phone_number_pattern = "[\+\d]?(\d{2,3}[-\.\s]??\d{2,3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})"
    res = re.sub(validate_phone_number_pattern, '<phone>', res)

    #email
    validate_email_pattern = "[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    res = re.sub(validate_email_pattern, '<email>', res)

    # tweet user mention regex
    validate_user_pattern = "@[a-zA-Z0-9_.+-]+"
    res = re.sub(validate_user_pattern, '@user', res)

    res=res.replace("@<email>", "@user")
    # res = p.tokenize(res)e
    return res


def preprocess_tweet_post(text=None):
    new_text = []
    for t in text.split(" "):
        if t.strip().lower() in ["@cnnbrk", "@cdcgov", "@bbcbreaking"] or t.strip().startswith('http'):
            continue
        t = '@user' if t.strip().startswith('@') else t # and len(t) >= 1
        # t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def both_contains(l1, l2, phrases):
    for phrase in phrases:
        if phrase in l1.lower() and phrase in l2.lower():
            return True
    return False
    return phrase in l1 and phrase in l2

def convert_sentint2sent(int_list):
    res=[]
    for i in int_list:
        if 0<=i<3:
            res.append(0)
        elif i==3:
            res.append(1)
        elif 3<i<=6:
            res.append(2)
        else:
            raise ValueError("int_list should be 0~6")
    return res

def convert_sentint2int(int_list):
    res=[]
    for i in int_list:
        res.append(abs(i-3))
    return res

# def convert_sentint2int(int_list):
#     res=[]
#     for i in int_list:
#         if 0<=i<3:
#             res.append(0)
#         elif i==3:
#             res.append(1)
#         elif 3<i<=6:
#             res.append(2)
#         else:
#             raise ValueError("int_list should be 0~6")
#     return res




def preprocess_tweet_local(text=None):
    new_text = []
    for t in text.split(" "):
        if t.lower() in ["@cnnbrk", "@cdcgov", "@bbcbreaking"] or t.startswith('http'):
            continue
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def find_URLS(string):
    # findall() has been used
    # with valid conditions for urls in string
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.findall(regex, string)
    return [x[0] for x in url]

def convert_list_to_dict(dic, key="id"):
    return {item[key]: item for item in dic}


def contains_category(reply, category="politics"):
    if "context_annotations" not in reply.keys(): return False

    DELETE_CONTEXT = ["Politicians", "Political Race", "Political Body"]
    DELETE_ID = ["35", "38", "88"]
    context_list = reply["context_annotations"]
    for context_item in context_list:
        if context_item["domain"]["name"] in DELETE_CONTEXT or context_item["domain"]["id"] in DELETE_ID:
            return True
    return False

def list_of_dicts_to_dict(list_of_dicts, key):
    return {item[key]: item for item in list_of_dicts}



def split_to_tr_val_test_data(data, ratio="811"):
    # f = load_file_lines(path)
    f = data
    tmp = list(ratio)
    assert len(tmp) == 3
    tr_ratio, dev_ratio, te_ratio = map(lambda d: int(d) / 10, tmp)
    assert tr_ratio + dev_ratio + te_ratio == 1
    # print("tr_ratio, dev_ratio, te_ratio ", tr_ratio, dev_ratio, te_ratio)

    tr, others = train_test_split(f, test_size=1 - tr_ratio)
    dev, te = train_test_split(others, test_size=te_ratio / (dev_ratio + te_ratio))

    # np.random.shuffle(f)
    #
    # tmp = list(ratio)
    # assert len(tmp) == 3
    # tr_ratio, dev_ratio, te_ratio = map(lambda d: int(d) / 10, tmp)
    #
    # print("tr_ratio, dev_ratio, te_ratio ",tr_ratio, dev_ratio, te_ratio)
    #
    # mid1, mid2 = int(tr_ratio * len(f)), int((tr_ratio+dev_ratio) * len(f))

    return tr, dev, te
# def validate_text(text):
#     text=text.strip()
#     if text in STOPWORDS:
#         return False
#     return True

class Printer:
    def __init__(self):
        self.cur_string=""
    def print(self, *strs):
        print(*strs)
        self.cur_string+=" ".join(strs)


def batchify(x, bs):
    return [x[i:i+bs] for i in range(0, len(x), bs)]
def check_data_unique(samples, keys):
    for key in keys:
        tmp=set()
        for sample in samples:
            if sample[key] in tmp:
                print(f"same {key} in data")
                embed()
                breakpoint()
            tmp.add(sample[key])
    # assert set([sample["title"] for sample in samples])==len(samples), breakpoint()

def construct_cache_name(name, *args, **kwargs):
    return name + "_" + "_".join(map(str, args)) + "_" + "_".join(map(str, kwargs.items()))

def get_samples_by_indices(samples,indices):
    return [samples[i] for i in indices]

def get_class_to_indices(samples, key='categories', filter_out_multiclass=False):
    categories_by_size = Counter([subitem for item in samples for subitem in item[key]])
    print("categories_by_size", categories_by_size)
    categories_by_size = sort_key_by_value(categories_by_size, reverse=True)  # largest left
    category2indices = {}
    multi_categories_indicies = set([i for i, sample in enumerate(samples) if len(sample[key])!=1])

    recorded_indicies = set()
    for k in categories_by_size:

        vals = [i for i, sample in enumerate(samples) if (i not in recorded_indicies) and
                ((i not in multi_categories_indicies) or (not filter_out_multiclass))]
        if vals:
            recorded_indicies.update(vals)
            category2indices[k] = vals
    return category2indices

def convert_dic_set_to_list(dic, do_sort=False):
    return {key: list(val) for key, val in dic.items()}



def stratified_split(data_or_filename, label_key=None, ratio=(.8, .1, .1), multilabel=False, save_file=False):

    assert np.sum(ratio) == 1, breakpoint()

    if save_file:
        data=load_file(data_or_filename)
    else:
        data=data_or_filename


    X=data
    y=np.array([item[label_key] for item in data])
    msss = StratifiedShuffleSplit(n_splits=1, test_size=1-ratio[0], train_size=ratio[0], random_state=0)
    train_index, test_index = list(msss.split(X, y))[0]
    X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
    y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]

    if ratio[2]<=0:
        return sorted(X_train.tolist()), sorted(X_test.tolist())

    test_ratio = ratio[2] / (ratio[1] + ratio[2])
    msss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, train_size=1-test_ratio, random_state=0)
    dev_index, test_index = list(msss.split(X_test, y_test))[0]
    X_dev, X_test_tmp = [X_test[i] for i in dev_index], [X_test[i] for i in test_index]
    X_test=X_test_tmp

    if save_file:
        path_name=get_path_name(data_or_filename)
        ext=get_ext(data_or_filename)
        dump_file_batch([X_train, X_dev, X_test], [path_name+split+ext for split in ["_train", "_dev", "_test"]])

    return X_train, X_dev, X_test



# category2indices no overlap
def _stratified_split(category2indices, ratio=(.8, .1, .1), id2cat=None, samples=None, multiclass=False):
    # ratio = (.8, .1, .1)

    X = np.array([0, 1, 2, 3, 4, 5, 6, 7, ])
    y = np.array([[0, 0], [0, 0], [0, 1], [0, 0], [1, 1], [0, 0], [1, 0], [1, 0]])

    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, train_size=.5, random_state=0)

    for train_index, test_index in msss.split(X, y):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]


    #accumulate
    min_ratio=min(ratio)
    ratio[1]+=ratio[0]
    # ratio[2]=1

    assigned_dict={}

    X_train, y_train, X_test, y_test = iterative_train_test_split(X, y, test_size=0.5)

    train_set, dev_set, test_set = set(), set(), set()
    for k in category2indices:
        cur_indices = category2indices[k]
        l=len(cur_indices)

        if int(len(cur_indices)*min_ratio) >= 1:
            np.random.shuffle(cur_indices)
            print("cur_indices", cur_indices)

            new_l=l+assigned_dict[k]["train"]+assigned_dict[k]["dev"]+assigned_dict[k]["test"]
            num_train, num_dev, num_test=int(new_l * ratio[0]), int(new_l * (ratio[1]-ratio[0])), int(new_l * ratio[2])
            num_train_additional, num_dev_additional, num_test_additional = num_train-assigned_dict[k]["train"], \
                                                                            num_dev - assigned_dict[k]["dev"], \
                                                                            num_test - assigned_dict[k]["test"]


            train_indices=cur_indices[:int(l * ratio[0])]
            dev_indices=cur_indices[int(l * ratio[0]):int(l * ratio[1])]
            test_indices=cur_indices[int(l * ratio[1]):]
            print("cur_indices", cur_indices)
            print("train_indices", train_indices)
            print("dev_indices", dev_indices)
            print("test_indices", test_indices)

            train_set.add(train_indices)
            dev_set.add(dev_indices)
            test_set.add(test_indices)


            for j in cur_indices:
                for cat in id2cat[j]:
                    if cat not in assigned_dict:
                        assigned_dict[cat]={"train":0, "dev":0, "test":0}
                    if j in train_set:
                        assigned_dict[cat]["train"]+=1
                    elif j in dev_set:
                        assigned_dict[cat]["dev"]+=1
                    elif j in test_set:
                        assigned_dict[cat]["test"]+=1
    return list(train_set), list(dev_set), list(test_set)

def sort_key_by_value(arr,v_key=None,  reverse=True):
    if v_key is not None:
        return [k for k, v in sorted(arr.items(), key=lambda x: x[1][v_key], reverse=reverse)]
    return [k for k, v in sorted(arr.items(), key=lambda x: x[1], reverse=reverse)]

def request_get(url, headers=None, params=None):
    if headers is None:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/39.0.2171.95 Safari/537.36'}
    try:
        # print("START REQUEST")
        time.sleep(0.2)
        if params is not None:
            r = requests.get(url, headers=headers, params=params)
        else:
            r = requests.get(url, headers=headers)

        # print("GOT")
        if r.ok:
            # print(r)
            return r
        else:

            print(r)
            return None
    except Exception as e:
        print(e)
        return None


def visualize_plot(x=None, y=None, label_names=None, path="", x_name="", y_name="", x_int=True, title=""):
    for i, sub_y in enumerate(y):
        plt.plot(range(len(sub_y)) if not x else x[i], sub_y, 'o-', label=label_names[i])
    if x_int:
        new_list = range(math.floor(min(x[0])), math.ceil(max(x[0])) + 1)
        plt.xticks(new_list)
    plt.legend()
    plt.title(title)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    if path: plt.savefig(path)
    plt.show()
    plt.clf()

def visualize_plot2(x=None, y=None, label_names=None, path="", x_name="", y_name="", x_int=True, title="", y_range=None, ax=None,axhline=None,axhline_name=None, size=None,fontsize=None):
    if ax is not None:
        plt=ax
    # for i, sub_y in enumerate(y):
    #     plt.plot(range(len(sub_y)) if not x else x, sub_y, 'o-', label=label_names[i])

    colors = ['tab:blue', 'tab:orange']  # Colorblind-friendly palette
    for i, sub_y in enumerate(y):
        plt.plot(range(len(sub_y)) if not x else x, sub_y, color=colors[i],
                 marker='o', linestyle='-', label=label_names[i])  # Add markers

    if x_int:
        new_list = range(math.floor(min(x[0])), math.ceil(max(x[0])) + 1)
        plt.xticks(new_list)

    plt.set_title(title, fontsize=size)
    if y_range: plt.set_ylim(y_range)
    plt.set_xlabel(x_name, fontsize=size)
    plt.set_ylabel(y_name, fontsize=size)

    if size:
        plt.xaxis.set_tick_params(labelsize=size)
        plt.yaxis.set_tick_params(labelsize=size)
        # if ax is None: plt.rcParams.update({'font.size': size})

    if axhline is not None:
        plt.axhline(axhline, linestyle='--', label=axhline_name)
        label_names+=[axhline_name]
    print(label_names)
    handles, _ = ax.get_legend_handles_labels()
    plt.legend(handles=handles[:], labels=label_names,fontsize=size-3)
    # plt.legend( fontsize=size) [:]

    if ax is not None:
        return

    if path: plt.savefig(path)
    plt.show()
    plt.clf()


def visualize_plot_multiple(x=None, y=None, label_names=None, path="", x_name="", y_name="", x_int=True, title="", y_range=None):
    pass



def modify_dict_keys(d, prefix=""):
    return {f"{prefix}{k}": v for k, v in d.items()}

def get_attribute_from_data(data, attribute, return_set=False, indices=None, condition_key=None, condition_key_2nd=None,condition_val=False):

    if condition_key_2nd is not None:

        key1, key2 = condition_key_2nd.split()
        res=[]
        for i,it in enumerate(data):
            if attribute in it:
                if (condition_key is not None and data[i][condition_key] == condition_val) or condition_key is None:
                    res.append(it[key1][key2])
            else:
                print("sample",i," does not have attribute", attribute)
                # print(it)
    else:
        if indices is not None:
            if condition_key is not None :res = [data[i][attribute] for i in indices if data[i][condition_key] == condition_val]
            else: res = [data[i][attribute] for i in indices if data[i][condition_key] == condition_val]
        else:
            res=[]
            for i,it in enumerate(data):
                if attribute in it:
                    if (condition_key is not None and data[i][condition_key] == condition_val) or condition_key is None:
                        res.append(it[attribute])
                else:
                    print("sample",i," does not have attribute", attribute)
                    # print(it)

    if return_set: res=set(res)
    return res


# def get_attribute_from_data(data, attribute, return_set=False, indices=None, condition_key=None, condition_val=False):
#     if indices is not None:
#         if condition_key is not None :res = [data[i][attribute] for i in indices if data[i][condition_key] == condition_val]
#         else: res = [data[i][attribute] for i in indices if data[i][condition_key] == condition_val]
#     else: res=[it[attribute] for it in data]
#
#     if return_set: res=set(res)
#     return res


def get_statistics(data_list):
    return {
        "min": min(data_list),
        "max": max(data_list),
        "mean": np.mean(data_list),
        "median": np.median(data_list),
        "std": np.std(data_list),
        "75%": np.percentile(data_list, 75),
        "25%": np.percentile(data_list, 25),
        "10%": np.percentile(data_list, 10),

    }


# def filter_data(data, attribute, return_set=False, indices=None):
#     if indices is not None:
#         res = [data[i][attribute] for i in indices]
#     else: res=[it[attribute] for it in data]
#
#     if return_set: res=set(res)
#     return res

def is_symmetric(g):
    return np.sum(np.abs(g.T - g)) == 0


def join(str1, str2):
    return os.path.join(str1, str2)


def get_ext(filename):
    return os.path.splitext(filename)[1]


def get_path_name(filename):
    return os.path.splitext(filename)[0]

def get_directory(path):
    return os.path.join(*(path.replace('\\', '/').split('/')[:-1]))


def get_filename_with_ext(filename):
    return os.path.split(filename)[-1]

def dump_file(obj, filename):
    if get_ext(filename) == ".json":
        with open(filename, "w+") as w:
            json.dump(obj, w)
    elif get_ext(filename) == ".pkl":
        with open(filename, "wb+") as w:
            pkl.dump(obj, w)
    elif get_ext(filename) == ".npy":
        with open(filename, "wb+") as w:
            np.save(w,obj)
    else:
        # print("not pkl or json")
        with open(filename, "w+", encoding="utf-8") as w:
            w.write(obj)

def dump_file_batch(objs, filenames):
    for obj, filename in zip(objs, filenames):
        dump_file(obj, filename)
def find_all_pos(my_list, target):
    return [i for i, x in enumerate(my_list) if x == target]



def path_exists(path):
    return os.path.exists(path)


def load_file(filename, init=None):
    if not path_exists(filename):
        if init is not None:
            print(f"File {filename} doesn't exist, initializing")
            return init

    if get_ext(filename) == ".json":
        with open(filename, "r", encoding="utf-8") as r:
            res = json.load(r)
    elif get_ext(filename) == ".jsonl":
        with open(filename, "r", encoding="utf-8") as r:
            res = [json.loads(line) for line in r]
    elif get_ext(filename) == ".html":
        res = codecs.open(filename, 'r', encoding="utf-8")

    elif get_ext(filename) in [".pkl",".pk"]:
        with open(filename, "rb") as r:
            res = pkl.load(r)
    elif get_ext(filename) in [".csv"]:
        res=pd.read_csv(filename)
    elif get_ext(filename) in [".npy"]:
        try:
            res = np.load(filename, allow_pickle=True)
        except Exception as e:
            print(e)
            time.sleep(2) # Try to Solve Too Many Files Open
            res = np.load(filename, allow_pickle=True)
    elif get_ext(filename) in [".txt"]:
        try:
            with open(filename, "r", encoding="utf-8") as r:
                res = r.read()
        except Exception as e:
            print(e)
            time.sleep(2)
            with open(filename, "r", encoding="utf-8") as r:
                res = r.read()
    return res
def load_file_batch(filenames, init=None):
    return [load_file(filename, init) for filename in filenames]

def browse_folder(path):
    return os.listdir(path)

def load_file_default(filename, init=None):
    if not path_exists(filename):
        if init == "{}": return {}
        if init == "[]": return []
        if init == 0: return 0
        if init == "": return ""
        return None
    if get_ext(filename) == ".json":
        with open(filename, "r", encoding="utf-8") as r:
            res = json.load(r)
            # try:
            #     res = json.load(r)
            # except:
            #     print("here")
            #     res = [json.loads(line.strip()) for i, line in enumerate(r)]
            #     return res
            #     print(r)
    elif get_ext(filename) == ".pkl":
        with open(filename, "rb") as r:
            res = pkl.load(r)
    return res


def load_file_lines(filename):
    if get_ext(filename) == ".json":
        with open(filename, mode="r", encoding="utf-8") as fin:
            res = [json.loads(line.strip()) for i, line in enumerate(fin)]
    elif get_ext(filename) == ".pkl":
        with open(filename, "rb") as r:
            res = pkl.load(r)
    return res


def split_files(file_path, num_files=8):
    """
    Splits a file into multiple files of size split_size
    :param file_path:
    :param split_size:
    :return:
    """
    data= load_file(file_path)
    path_name, ext= os.path.splitext(file_path)
    split_size = math.ceil(len(data)/num_files)
    data_split = [data[(i*split_size):min((i+1)*split_size, len(data))] for i in range(0, num_files)]
    filenames = []
    for i, split in enumerate(data_split):
        filename = path_name + "_" + str(i) + ext
        dump_file(split, filename)
        filenames.append(filename)
    return filenames


def merge_files(base_file_path, num_files=8, save_data=False):
    """
    Merges multiple files into one file
    :param file_paths:
    :return:
    """
    path_name, ext= os.path.splitext(base_file_path)
    file_paths = [path_name + "_" + str(i) + ext for i in range(num_files)]
    data = []
    for file_path in file_paths:
        data += load_file(file_path)
    if save_data:dump_file(data, base_file_path)
    return data

def mkdir(dir):

    if not os.path.isdir(dir):
        os.mkdir(dir)


def module_exists(module_name):
    try:
        __import__(module_name)
    except ImportError:
        return False
    else:
        return True


# def get_gpu_mem_info():
#     nvmlInit()
#     h = nvmlDeviceGetHandleByIndex(0)
#     info = nvmlDeviceGetMemoryInfo(h)
#     print(f'total    : {info.total}')
#     print(f'free     : {info.free}')
#     print(f'used     : {info.used}')

def check_error(input_list):
    """
    check data quality
    :param input_list: list of instances
    :return: None
    """
    for item in input_list:
        sections=item['sections']
        if not len(sections):
            print(item)
            # breakpoint()
            embed()
        for section in sections:
            if not section:
                print(item)
                embed()
            for step in section:
                if not step:
                    print(item)
                    embed()

def check_error2(input_list):
    """
    check data quality
    :param input_list: list of instances
    :return: None
    """

    for item in input_list:
        src, tgt = item["src_text"], item['tgt_text']
        if not src.replace("[SEP]", "").strip() or not tgt.replace("[SEP]", "").strip():
            print(src, "\n", tgt)
            embed()
    print("ok")

def bool_flag(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def check_path(path):
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)


def get_path_info(path):
    path_name, ext = os.path.splitext(path)
    directory = os.path.join(*(path.replace('\\', '/').split('/')[:-1]))
    filename, _ = os.path.splitext(os.path.split(path)[-1])

    return directory, path_name, filename, ext

def get_best_ckpt():
    all_ckpts = list(glob("model/states/checkpoint-*"))
    all_ckpts_ids = np.array([int(item.split("checkpoint-")[-1]) for item in all_ckpts])
    best_ckpt = all_ckpts[all_ckpts_ids.argsort()[-1]]
    print("best_ckpt", best_ckpt)
    return best_ckpt
    # best_ckpt = sorted(list(glob("model/states/checkpoint-*")), reverse=True)[0]
    # print("all ckpts", sorted(list(glob("model/states/checkpoint-*")), reverse=True))
    # best_ckpt = "model/states/checkpoint-80000"

def check_file(file):
    return os.path.isfile(file)


def export_config(config, path):
    param_dict = dict(vars(config))
    check_path(path)
    with open(path, 'w') as fout:
        json.dump(param_dict, fout, indent=4)


def freeze_net(module):
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_net(module):
    for p in module.parameters():
        p.requires_grad = True


def test_data_loader_ms_per_batch(data_loader, max_steps=10000):
    start = time.time()
    n_batch = sum(1 for batch, _ in zip(data_loader, range(max_steps)))
    return (time.time() - start) * 1000 / n_batch