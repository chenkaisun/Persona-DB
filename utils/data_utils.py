import numpy as np
from torch.utils.data import Dataset

# from copy import deepcopy
# from datasets import load_dataset
# from transformers.data.data_collator import DataCollatorWithPadding
# from copy import deepcopy
# import csv
# import scipy
from utils.utils import *

# if module_exists("torch_geometric"):
#     from torch_geometric.data import Batch, Data
from copy import deepcopy, copy
from multiprocessing import Pool
from tqdm import tqdm
from collections import defaultdict
from dataclasses import dataclass

from typing import Any, List, Optional, Union

# import torch
from transformers.file_utils import PaddingStrategy
from transformers.modeling_utils import PreTrainedModel
# from transformers import PreTrainedTokenizerBase

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
# from nltk.corpus import stopwords
# nltk.download('stopwords')
from nltk.corpus import wordnet as wn

from nltk.stem import WordNetLemmatizer

import torch.nn.functional as F

from transformers import PreTrainedTokenizerBase, AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from constants import *
import random

import torch
import numpy as np
# from scipy.spatial.distance import cosine
from typing import List
if module_exists("faiss"):
    import faiss
from llm_api import chatbot
from sklearn.metrics.pairwise import cosine_similarity
import preprocessor as p

#
# def process_triplet(cur_struc_vec, cur_history_vec, cur_dict, refine_type="r"):
#     tmp_hist_vec = []
#     for k, v in cur_dict.items():
#         if any([key in k.lower() for key in desc_keys]):
#             if isinstance(v, list):
#                 cur_struc_vec.append(f"{k}: {', '.join(v)}")
#             elif isinstance(v, str):
#                 cur_struc_vec.append(f"{k}: {v}")
#             else:
#                 breakpoint()
#         elif any([key in k.lower() for key in list_keys]):
#             if not isinstance(v, list):
#                 breakpoint()
#             tmp_hist_vec.extend(v)
#         else:
#             # breakpoint()
#             continue
#     # tmp_hist_vec sort
#     tmp_hist_vec = sorted(tmp_hist_vec)
#     cur_history_vec.extend([(f"{user_id}_{refine_type}_{i}", item) for i, item in enumerate(tmp_hist_vec)])

import pandas as pd
import os
from datetime import datetime

# Function to generate filename
def generate_filename(case, prompt_type, neighbor_topk, item_topk, pure_intensity, version):
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"chatgpt_case{case}_{prompt_type}_neighbor{neighbor_topk}_item{item_topk}_pint{pure_intensity}_{version}_{current_time}.json"
    return filename

# Function to create or load the DataFrame
def load_or_create_dataframe(file_path,eval_keys):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        columns = ['Time', 'Case', 'PromptType', 'NeighborTopK', 'NeighborThreshold', 'ItemTopK','ItemTopKCollab', 'ItemThreshold', 'HistL', 'NumHist','IncludeHistory','IncludeEmptyNeighbors','DoProcess', 'PureIntensity','LCModel','NeighborHistOnly','Modules', 'IncludeRF', 'Version']+eval_keys
        return pd.DataFrame(columns=columns)
# Function to update and save the DataFrame
def update_dataframe(df, file_path, current_time, case, prompt_type, neighbor_topk, neighbor_threshold, item_topk, item_topk_collab, item_threshold, histl,n_hist,include_hist,include_empty_neighbors,do_process,pure_intensity, lc_model, nb_hist_only,modules,include_rf,version, eval_result):
    new_row = {
        "Time": current_time,
        'Case': case,
        'PromptType': prompt_type,
        'NeighborTopK': neighbor_topk,
        'NeighborThreshold': neighbor_threshold,
        'ItemTopK': item_topk,
        'ItemTopKCollab': item_topk_collab,
        'ItemThreshold': item_threshold,
        'HistL': histl,
        'NumHist': n_hist,
        'IncludeHistory': include_hist,
        'IncludeEmptyNeighbors': include_empty_neighbors,
        'DoProcess': do_process,
        'PureIntensity': pure_intensity,
        'LCModel':lc_model,
        'NeighborHistOnly': nb_hist_only,
        'Modules': modules,
        'IncludeRF': include_rf,
        'Version': version
    }
    # for k in eval_keys:
    #     new_row[k] = eval_result[k]
    new_row.update(eval_result)

    df = df.append(new_row, ignore_index=True)
    # print(df)

    return df



def get_data_slice(data, num_nodes, node_id):
    data = sorted(data)
    data = batchify(data, math.ceil(len(data) / num_nodes))
    if int(node_id) >= len(data):
        print("node_id >= the number of batches")
        exit()
    return data[node_id]


def fill_prompt(prompt, list_items, ids=None):
    if ids is not None: return [(i, prompt + item) for (i, item) in zip(ids, list_items)]
    return [(i, prompt + item) for i, item in enumerate(list_items)]


def standardize_keys(s):
    res = " ".join([word.capitalize() for word in s.split()])
    # make it plural

    if res.startswith("Ideolog"):
        res = "Ideologies"
    else:
        if res.endswith("s"):
            res = res[:-1]
        res = res + "s"

    return res


def remove_dup(vec):
    # remove the ones where the same text is repeated for second position in tuple
    seen = set()
    res = []
    for item in vec:
        key=item[1].lower()
        if key not in seen:
            seen.add(key)
            res.append(item)
    return res

def process_triplet(user_id, cur_struc_vec, cur_history_vec, cur_dict, refine_type="r", do_process=False):
    tmp_hist_vec = []
    for k, v in cur_dict.items():
        if any([key in k.lower() for key in desc_keys]):
            if isinstance(v, list):
                cur_struc_vec.append(f"{standardize_keys(k)}: {', '.join(v)}")
            elif isinstance(v, str):
                cur_struc_vec.append(f"{standardize_keys(k)}: {v}")
            elif isinstance(v, dict):
                print("is a dict in desc_keys")
            else:
                print("in desc_keys")
                breakpoint()
        elif any([key in k.lower() for key in list_keys]):
            if not isinstance(v, list):
                print("in list_keys")
                breakpoint()

            tmp = [item for item in v if isinstance(item, str)]

            ##### Maybe remove view toward:None
            if len(tmp)>1 and not tmp[0].strip().startswith("View toward") and " - " in tmp[0] and " - " in tmp[1]:
                tmp=[f"View toward {st}" for st in tmp]
            if do_process:
                # print("do_process")
                try:
                    tmp = [item.strip() for item in tmp if item.strip().lower().startswith("view toward") or item.strip().lower().startswith("the user")]
                    tmp = [item+"." if not item.endswith(".") else item for item in tmp]
                except:
                    print("except")
                    breakpoint()
            tmp_hist_vec.extend(tmp)
            # if all([isinstance(item, str) for item in v]):
            #     tmp_hist_vec.extend(v)
        else:
            # breakpoint()
            continue
    # tmp_hist_vec sort
    try:
        if do_process: tmp_hist_vec = set(tmp_hist_vec)
        tmp_hist_vec = sorted(tmp_hist_vec)  # set
    except Exception as e:
        print(e)
        print("tmp_hist_vec", tmp_hist_vec)
        breakpoint()


    # if len(set(tmp_hist_vec)) != len(tmp_hist_vec):
    #     # print("duplicate", user_id, refine_type)
    #     # print("difference lens are", abs(len(set(tmp_hist_vec))-len(tmp_hist_vec)))
    #     if abs(len(set(tmp_hist_vec))-len(tmp_hist_vec))>1000:
    #         print("too many duplicates", user_id, refine_type)
    #         # breakpoint()
    #     pass
    cur_history_vec.extend([(f"{user_id}_{refine_type}_{i}", item) for i, item in enumerate(tmp_hist_vec)])



class UserRetriever:
    def __init__(self, args=None, user_data=None, user_desc=None):
        # self.all_uids = set(load_file(CNN_users_info_rf_filename))
        # get all filename under CNN_r_anno_json_dir
        self.args = args
        if self.args.dataset != "CNN":
            CNN_dir = f"{data_dir}OpinionQA/human_resp/American_Trends_Panel_W{self.args.dataset}/"
            CNN_r_anno_json_dir = f"{CNN_dir}chatgptanno_r_json/"
            CNN_rf_anno_json_dir = f"{CNN_dir}chatgptanno_rf_json/"
            CNN_non_empty_hist_uids_filename=""
            CNN_non_empty_hist_uids_processed_filename=""

        self.user_data = user_data
        self.user_desc = user_desc
        if user_data is not None:
            self.all_uids = set(user_data.keys())
        else:
            self.all_uids = {uid.split(".")[0] for uid in os.listdir(CNN_r_anno_json_dir)}

        self.uid2analysis = {}
        self.uid2profile = {}
        self.layer_modules= args.modules
        print("layer_modules", self.layer_modules)

        if args is not None:
            self.do_process = args.do_process
            if path_exists(CNN_non_empty_hist_uids_filename) and path_exists(CNN_non_empty_hist_uids_processed_filename):
                self.non_empty_hist_uids = set(load_file(CNN_non_empty_hist_uids_filename if not args.do_process else
                                                     CNN_non_empty_hist_uids_processed_filename))
            else:
                self.non_empty_hist_uids = self.all_uids

        else:
            self.do_process = False
            self.non_empty_hist_uids = set(load_file(CNN_non_empty_hist_uids_filename))

        # for uid in self.all_uids:
        #     self.get_user_profile(uid)
        #     self.get_user_analysis(uid)

    def get_user_profile(self, uid, data=None):

        if uid in self.uid2profile:
            return self.uid2profile[uid]

        if data is not None:
            # print("data passed")
            self.uid2profile[uid] = data
            return self.uid2profile[uid]


        if self.args.dataset != "CNN":
            CNN_r_anno_json_dir = f"{CNN_dir}chatgptanno_r_json/"

        user_profile = load_file(f"{CNN_r_anno_json_dir}{uid}.json", init={})
        cur_struc_vec = []
        for k, v in user_profile.items():
            if any([key in k.lower() for key in user_meta_keys]):
                if isinstance(v, list):
                    cur_struc_vec.append(f"{standardize_keys(k)}: {', '.join(v)}")
                elif isinstance(v, str):
                    cur_struc_vec.append(f"{standardize_keys(k)}: {v}")
                elif isinstance(v, dict):
                    print("is a dict in get_user_profile desc_keys")
                else:
                    print("in get_user_profile desc_keys")
                    breakpoint()
        self.uid2profile[uid] = ("\n".join(cur_struc_vec))
        return self.uid2profile[uid]

    def get_user_analysis(self, uid, nb_hist_only=False): #, hist_only=False

        if uid in self.uid2analysis:
            return self.uid2analysis[uid]
        # , init = {}
        if nb_hist_only:
            #include current query as well
            cur_history_vec=[(f"{uid}_hist_{i}", item[1]) for i, item in enumerate(self.user_data[uid])]
            self.uid2analysis[uid] = cur_history_vec
            return cur_history_vec

        if self.args.dataset != "CNN":
            CNN_dir = f"{data_dir}OpinionQA/human_resp/American_Trends_Panel_W{self.args.dataset}/"
            CNN_r_anno_json_dir = f"{CNN_dir}chatgptanno_r_json/"
            mkdir(CNN_r_anno_json_dir)
            CNN_rf_anno_json_dir = f"{CNN_dir}chatgptanno_rf_json/"
        cur_dict_r, cur_dict_rf = load_file(f"{CNN_r_anno_json_dir}{uid}.json", init={}), load_file(f"{CNN_rf_anno_json_dir}{uid}.json", init={})
        cur_struc_vec, cur_history_vec = [], []
        # =====don't include neighbors that have empty refined vecs+=======
        # visited=set()

        for cur_d, refine_type in zip([cur_dict_r, cur_dict_rf], ["r", "rf"]):

            if refine_type not in self.layer_modules:
                continue

            tmp_hist_vec = []
            for k, v in cur_d.items():
                if any([key in k.lower() for key in desc_keys]) and any([key in k.lower() for key in list_keys]):
                    print("both desc_keys and list_keys")
                    # breakpoint()
                if any([key in k.lower() for key in desc_keys]):
                    pass
                elif any([key in k.lower() for key in list_keys]):
                    # if not isinstance(v, list):
                    #     breakpoint()
                    # tmp_hist_vec.extend(v)
                    if not isinstance(v, list):
                        print("in list_keys")
                        breakpoint()
                    tmp = [item for item in v if isinstance(item, str)]
                    if len(tmp) > 1 and not tmp[0].strip().startswith("View toward") and " - " in tmp[0] and " - " in tmp[1]:
                        tmp = [f"View toward {st}" for st in tmp]
                    if self.do_process:
                        tmp = [item.strip() for item in tmp if item.strip().lower().startswith("view toward") or item.strip().lower().startswith("the user")]
                        tmp = [item+"." if not item.endswith(".") else item for item in tmp]
                        # tmp2=[]
                        # for item in tmp:
                        #     if item in visited or not (item.strip().lower().startswith("view toward") or item.strip().lower().startswith("the user")):
                        #         continue
                        #     visited.add(item)
                        #     tmp2.append(item)
                        # tmp=tmp2

                    tmp_hist_vec.extend(tmp)
                else:
                    continue

            if self.do_process: tmp_hist_vec = set(tmp_hist_vec)
            tmp_hist_vec = sorted(tmp_hist_vec)

            # if len(set(tmp_hist_vec)) != len(tmp_hist_vec):
            #     print("duplicate", uid, refine_type)
            cur_history_vec.extend([(f"{uid}_{refine_type}_{i}", item) for i, item in enumerate(tmp_hist_vec)])

        # rf and r might overlap

        self.uid2analysis[uid] = cur_history_vec
        return cur_history_vec

    def retrieve_users(self, uid, top_k=2, include_empty_neighbors=False): #uids=None,
        # call retrieve api embedding to get top neighbors

        if self.args.dataset != "CNN":
            CNN_dir = f"{data_dir}OpinionQA/human_resp/American_Trends_Panel_W{self.args.dataset}/"
            CNN_api_embeddings_dir = f"{CNN_dir}api_embeddings/"

        neighbor_ids_dir = f"{CNN_api_embeddings_dir}user_meta_key_retrieval_uids_top100_dp{self.do_process}/"
        if include_empty_neighbors:
            neighbor_ids_dir = f"{CNN_api_embeddings_dir}user_meta_key_retrieval_uids_top100_en1/"
        mkdir(neighbor_ids_dir)
        if path_exists(f"{neighbor_ids_dir}{uid}.json"):
            return load_file(f"{neighbor_ids_dir}{uid}.json")[:top_k]

        curr_user_profile = self.get_user_profile(uid, self.user_desc[str(uid)]["description"] if self.user_desc is not None else None)
        neighbor_uids = list(self.all_uids - {uid})
        if not include_empty_neighbors:
            # non_empty_hist_uids = load_file(CNN_non_empty_hist_uids_filename)
            # list(set(neighbor_uids) & self.non_empty_hist_uids)
            neighbor_uids =list(self.non_empty_hist_uids - {uid})

        # print("e5")
        try:
            neighbor_profiles = [self.get_user_profile(cur_uid, self.user_desc[cur_uid]["description"] if self.user_desc is not None else None) for cur_uid in neighbor_uids]
        except:
            print("error in get_user_profile")
            embed()
        # print("e6")

        # filter neighbor_uids and neighbor_profiles if neighbor_profile is empty
        neighbor_uids, neighbor_profiles = zip(*[(cur_uid, cur_profile) for cur_uid, cur_profile in zip(neighbor_uids, neighbor_profiles) if cur_profile.strip()])
        neighbor_uids, neighbor_profiles = list(neighbor_uids), list(neighbor_profiles)

        doc_prompt = "Encode the analysis on the user for retrieval. This analysis on the user may hint at the user's opinions and personality. Consider the analysis carefully. Analysis: \n"
        query_prompt = "Encode this analysis to find the most relevant user analysis. This analysis on the user may hint at the user's opinions and personality. Consider the analysis carefully. Analysis: \n"
        query, database = curr_user_profile, neighbor_profiles
        query = fill_prompt(query_prompt, [query], ids=[uid])[0]
        database = fill_prompt(doc_prompt, database, ids=neighbor_uids)
        # print(f"\nretrieve top neighbors\n")
        # print("e7")

        # run_name = f"user_meta_key_retrieval_en0_dp{self.do_process}" #if not self.do_process else "user_meta_key_retrieval_do_process"
        # if include_empty_neighbors: run_name = "user_meta_key_retrieval"
        run_name = "user_meta_key_retrieval"
        top_idxs = retrieve_api_embedding(query, database, top_k=100, run_name=run_name, save=True, dataset=self.args.dataset)
        res=[neighbor_uids[i] for i in top_idxs]
        dump_file(res, f"{neighbor_ids_dir}{uid}.json")
        top_k_res = res[:top_k]

        # print(f"\n[retrieved_uids]\n{[int(i) for i in top_k_res]}\n")
        # print(f"[curr_user_profile]\n{curr_user_profile}")
        # print("\n[curr_neightbor_profile]")
        # print("\n\n".join([neighbor_profiles[i] for i in top_idxs[:top_k]]))
        # print()

        return top_k_res

    def join_db(self, uids, nb_hist_only=False):
        # for uid in uids:
        #     visited={}
        #     for element in self.get_user_analysis(uid):
        #         if element[1].strip() in visited:
        #             breakpoint()
        #         visited[element[1].strip()]=1
        #
        # for uid in uids:
        #
        #     refined_history_vec = [element for uid in uids for element in self.get_user_analysis(uid)]
        return [element for uid in uids for element in self.get_user_analysis(uid, nb_hist_only)]


def get_user_profile(user_id, user_data, history_data, top_n_hist=50):
    """user_desc"""
    if str(user_id) not in user_data:
        user_desc = ""
        print("str(user_id) not in user_data:str(user_id) not in user_data")
        # continue
    else:
        user_desc = user_data[str(user_id)]['description']

    """history"""
    if str(user_id) not in history_data:
        history_text = ""
        history_posts = []
        history_tweet_ids = []
        # continue
    else:
        history_posts = [preprocess_tweet_link(item["text"]).strip() for item in history_data[str(user_id)][:top_n_hist]]
        history_tweet_ids = [item["tweet_id"] for item in history_data[str(user_id)][:top_n_hist]]
        history_text = ";".join(history_posts)
    user_desc = user_desc.strip()
    history_text = history_text.strip()
    # if not user_desc.strip() and not history_text.strip():
    #     return None, None
    return user_desc, history_text, history_posts, history_tweet_ids


def get_user_profile2(user_id, user_data, history_data, top_n_hist=50,post_id=None):
    "opinonQA"

    """user_desc"""
    if str(user_id) not in user_data:
        user_desc = ""
        print("str(user_id) not in user_data:str(user_id) not in user_data")
        # continue
    else:
        user_desc = user_data[str(user_id)]['description']

    """history"""
    if str(user_id) not in history_data:
        history_text = ""
        history_posts = []
        history_tweet_ids = []
        # continue
    else:
        history_posts = [item[1].strip() for i,item in enumerate(history_data[str(user_id)][:top_n_hist]) if item[0]!=post_id]
        history_tweet_ids = [str(i) for i,item in enumerate(history_data[str(user_id)][:top_n_hist]) if item[0]!=post_id]
        history_text = ";".join(history_posts)
    user_desc = user_desc.strip()
    history_text = history_text.strip()
    # if not user_desc.strip() and not history_text.strip():
    #     return None, None
    return user_desc, history_text, history_posts, history_tweet_ids



def get_user_profile3(user_id, user_data, history_data, top_n_hist=50,uid_to_excluded_post_ids=None):
    "opinonQA"

    """user_desc"""
    if str(user_id) not in user_data:
        user_desc = ""
        print("str(user_id) not in user_data:str(user_id) not in user_data")
        # continue
    else:
        user_desc = user_data[str(user_id)]['description']

    """history"""
    if str(user_id) not in history_data:
        history_text = ""
        history_posts = []
        history_tweet_ids = []
        # continue
    else:
        history_posts = [item[1].strip() for i,item in enumerate(history_data[str(user_id)][:top_n_hist]) if item[0] not in uid_to_excluded_post_ids[user_id]]
        history_tweet_ids = [str(i) for i,item in enumerate(history_data[str(user_id)][:top_n_hist]) if item[0] not in uid_to_excluded_post_ids[user_id]]
        history_text = "\n".join(history_posts)
    user_desc = user_desc.strip()
    history_text = history_text.strip()
    # if not user_desc.strip() and not history_text.strip():
    #     return None, None
    return user_desc, history_text, history_posts, history_tweet_ids



# get chatgpt response prediction

def parse_gpt_anno2(s):
    """parse the annotation from structurized and high-level"""

    pass


def retrieve_api_embedding(query, database, top_k=8, run_name="test", save=False, q_ids=None, doc_ids=None, query_prompt="", doc_prompt="", threshold=0, dataset="CNN", uid=0):
    # def fill_prompt(prompt, list_items, ids=None):
    #     if ids is not None: return [(i, prompt + item) for (i, item) in zip(ids, list_items)]
    #     return [(i, prompt + item) for i, item in enumerate(list_items)]
    #
    # query = fill_prompt(query_prompt, [query], ids=q_ids)[0]
    # database = fill_prompt(doc_prompt, database, ids=doc_ids)
    if not database: return []
    # print("\nStarting retrieve_api_embedding\n")

    if dataset!="CNN":
        CNN_dir = f"{data_dir}OpinionQA/human_resp/American_Trends_Panel_W{dataset}/"
        CNN_api_embeddings_dir = f"{CNN_dir}api_embeddings/"

    q_dir, doc_dir = f"{CNN_api_embeddings_dir}{run_name}_query/", f"{CNN_api_embeddings_dir}{run_name}_doc/"
    mkdir(CNN_api_embeddings_dir)
    mkdir(q_dir)
    mkdir(doc_dir)

    qid, query = query
    q_filename = f"{q_dir}/{qid}.npy"
    if not path_exists(q_filename) or not save:
        query_embeddings = np.array(chatbot([query], "embedding"))
        dump_file(query_embeddings, q_filename)
    query_embeddings = load_file(q_filename)

    res = []

    doc_ids_to_be_done = [(doc_id, doc) for (doc_id, doc) in database
                          if not path_exists(f"{doc_dir}/{doc_id}.npy")] if save else database
    if len(database) and len(doc_ids_to_be_done) or not save:  # not path_exists(f"{doc_dir}/{database[-1][1]}.npy")
        # print("\nGenerate embeddings\n")
        doc_embs = chatbot([doc for (doc_id, doc) in doc_ids_to_be_done], "embedding")
        # doc_embs = [chatbot([doc], "embedding")[0] for doc_id, doc in database]
        for i, (doc_id, doc) in enumerate(doc_ids_to_be_done):
            doc_filename = f"{doc_dir}/{doc_id}.npy"
            emb = doc_embs[i]  # chatbot(doc, "embedding")
            dump_file(emb, doc_filename)
    for i, (doc_id, doc) in enumerate(database):
        doc_filename = f"{doc_dir}/{doc_id}.npy"
        emb = load_file(doc_filename)
        res.append(emb)
    # print("\nCompute similarity")
    corpus_embeddings = np.array(res)
    try:
        similarities = cosine_similarity(query_embeddings, corpus_embeddings)
    except:
        breakpoint()
    # similarities = cosine_similarity(query_embeddings, corpus_embeddings)

    # print("Finished computing similarity")
    # Sort the similarity scores in descending order to get the most similar documents first
    sorted_indices = np.argsort(similarities, axis=1)[:, ::-1]
    # print("Finished sorting")
    # print(sorted_indices)
    # Assuming you want to retrieve the top N most similar documents (adjust N as needed)

    # Get the top N document indices
    top_N_indices = sorted_indices[:, :top_k][0]

    # get top k similarity scores
    top_N_scores = similarities[:, top_N_indices][0]
    # print("Finished get top k similarity scores")

    # onlt return top the top k ones that have similarity score above threshold
    if threshold>0:
        # print("threshold", threshold)
        top_N_indices = top_N_indices[top_N_scores>threshold]
        # top_N_scores = top_N_scores[top_N_scores>threshold]
    # print top entries
    # print([(database[i], top_N_scores[i]) for i in range(len(top_N_indices))])

    # print(top_N_scores)
    return top_N_indices


def load_all_data(load_history=False):
    train_anno, dev_anno, test_anno = load_file_batch([CNN_train_anno_filename, CNN_dev_anno_filename, CNN_test_anno_filename])
    train, dev, test = train_anno, dev_anno, test_anno
    post_data, user_data = load_file_batch(filenames=[CNN_post_filename, CNN_users_info_rf_filename])
    users_followings_dict = load_file(CNN_users_followings_dict_filename)
    history_data = None
    if load_history: history_data = load_file(CNN_users_history_filename)
    random.seed(0)
    np.random.seed(0)
    return train, dev, test, train_anno, dev_anno, test_anno, post_data, user_data, users_followings_dict, history_data


# def get_consine_sim_matrix(a,b):
#     a_norm = a / a.norm(dim=1)[:, None]
#     b_norm = b / b.norm(dim=1)[:, None]
#     res = torch.mm(a_norm, b_norm.transpose(0, 1))
#     # print(res)
#     return res
class GlobalIdMap:
    def __init__(self):
        self.global_user_id_map = {}
        self.global_user_id_map_rev = {}
        self.global_user_id = 0

    def insert_id(self, uid):
        if uid in self.global_user_id_map:
            return self.global_user_id_map[uid]
        self.global_user_id_map.setdefault(uid, self.global_user_id)
        self.global_user_id_map_rev.setdefault(self.global_user_id, uid)
        self.global_user_id += 1
        return self.global_user_id_map[uid]

    def get_id(self, uid):
        return self.global_user_id_map[uid] if uid in self.global_user_id_map else None

    def get_rev_id(self, uid):
        return self.global_user_id_map_rev[uid] if uid in self.global_user_id_map_rev else None

    def get_map(self):
        return self.global_user_id_map

    def get_size(self):
        return len(self.global_user_id_map)

    def get_keys_sorted_by_values(self):
        res = []
        for i in range(self.get_size()):
            res.append(self.get_rev_id(i))
        return res

    def __getitem__(self, uid):
        return self.insert_id(uid)


def preprocess_tweet_link(text=None):
    new_text = []
    for t in text.split():
        if t.startswith('http'):
            continue
        new_text.append(t)
    return " ".join(new_text)


def load_response_data(load_history_data=False):  # , post, profile, history sample,

    train, dev, test = load_file_batch([CNN_train_filename, CNN_dev_filename, CNN_test_filename])
    train_anno, dev_anno, test_anno = load_file_batch([CNN_train_anno_filename, CNN_dev_anno_filename, CNN_test_anno_filename])
    post_data, user_data = load_file_batch(filenames=[CNN_post_filename, CNN_users_info_filename])
    users_followings_dict = load_file(CNN_users_followings_dict_filename)
    id_to_index = load_file(CNN_id_to_index)
    user_data = load_file(CNN_users_info_sc_filename)

    # average of len of values in users_followings_dict
    print(sum([len(v) for v in users_followings_dict.values()]) / len(users_followings_dict))

    history_data = None
    if load_history_data: history_data = load_file(CNN_users_history_filename)

    valid_uids = set()
    for data in [train, dev, test]:
        for item in data:
            valid_uids.add(item["author_id"])
    valid_uids_list = sorted(list(valid_uids))

    cnt = 0
    for uid in users_followings_dict:
        for nb_id in users_followings_dict[uid]:
            if str(nb_id) in id_to_index and str(uid) in id_to_index:
                cnt += 1
    return train, dev, test, train_anno, dev_anno, test_anno, post_data, user_data, users_followings_dict, id_to_index, history_data, valid_uids_list


def generate_debug_data():
    # train, dev, test, train_anno, dev_anno, test_anno, post_data, user_data, users_followings_dict, id_to_index, history_data, valid_uids_list = load_response_data(False)

    mkdir(f"{CNN_dir}debug/")

    #####  post data #####
    post_data = {}
    for i, txt in enumerate(["0", "1", "2"]):
        post_data[i] = {
            "conversation_id": i,
            "text": f"post {i}"
        }

    #####  user data #####
    user_data = {}
    history_data = {}

    for pos in range(6):
        user_data[pos] = {
            "description": f"desc {pos}",
            "category": "desc",
        }
        history_data[pos] = [
            {
                "text": f"hist {pos}",
                "category": "hist",
            }
        ]

    response_data = []
    pos = 0
    for i, post_item in post_data.items():
        for j, user_item in user_data.items():
            response_data.append({
                "conversation_id": str(i),
                "author_id": str(j),
                "text": f"response {pos} user {j} post {i}",
                "tweet_id": str(pos),
                "predicted": pos % 3,
                "predicted_intensity": pos % 7,
            })
            pos += 1
    # random social network
    users_followings_dict = {
        "0": [1, 2],
        "1": [0, 2],
        "2": [0, 1],
        "3": [1],
        "4": [],
        "5": [3],
    }
    id_to_index = {str(i): i for i in range(5)}

    mkdir(f"{CNN_dir}custom_files/")

    dump_file_batch([response_data, post_data, users_followings_dict, id_to_index, user_data, history_data], [f"{CNN_dir}debug/{fn}.json" for fn in [
        "response_data_balanced_train_anno", "post_data", "users_followings_dict", "id_to_index", "users_info_dict", "users_history_dict"
    ]])

    dump_file_batch([response_data, response_data], [f"{CNN_dir}debug/{fn}.json" for fn in [
        "response_data_balanced_test_anno", "response_data_balanced_dev_anno"
    ]])


def process_news_profile_history(history_data, user_data, post_data, user_id, post_id, tgt_text=None, num_past=50):  # , post, profile, history sample,
    """user_desc"""
    post_text, user_desc, history_text = None, None, None
    if user_id is not None:
        if str(user_id) not in user_data:
            user_desc = ""
            # continue
        else:
            user_desc = user_data[str(user_id)]['description']

        if str(user_id) not in history_data:
            history_text = ""
            # continue
        else:
            history_posts = [preprocess_tweet_link(item["text"]) for item in history_data[str(user_id)][:num_past]]
            history_text = ";".join(history_posts)

        if not user_desc.strip() and not history_text.strip():
            user_desc = history_text = None
    if post_id is not None:
        # filter out inactive users
        post_text = post_data[str(post_id)]['text']
        post_text = preprocess_tweet_local(post_text)
        if not post_text.strip():
            post_text = None
    if tgt_text is not None:
        if find_URLS(tgt_text):
            tgt_text = None
        else:
            tgt_text = p.clean(tgt_text)
            if not tgt_text.strip():
                tgt_text = None
    return post_text, user_desc, history_text, tgt_text


# def filter_dataset(original_data, history_data, user_data, post_data, use_gnn=1, case_study=0, cur_split=""):
#     p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.HASHTAG, p.OPT.RESERVED, p.OPT.EMOJI, p.OPT.SMILEY)
#     res_new = []
#     case_study_dict = load_file(f'{CNN_dir}case_study_dict.json') if case_study == 2 else {}
#
#     for i, sample in enumerate(tqdm(original_data)):  # this indicates ith path after breaking out all articles into individual paths
#         user_id = sample["author_id"]
#         post_id = sample["conversation_id"]
#
#         if case_study == 2:
#             if (cur_split == "dev" and user_id not in case_study_dict["unseen_dev"]) or (cur_split == "test" and user_id not in case_study_dict["unseen_test"]):
#                 continue
#
#         """user_desc"""
#         if str(user_id) not in user_data:
#             user_desc = ""
#             # continue
#         else:
#             user_desc = user_data[str(user_id)]['description']
#
#         # if not user_desc.strip():
#         #     num_empty_desc += 1
#         #     if args.skip_empty_profile: continue
#
#         # pretrain
#         # in reply to account id, author id, use for recording conversation
#         # geo
#         # timestamp
#
#         "only use nonreply"
#
#         # history
#         # geo TODO
#         # entities/context
#         # cat top 20, max is max seq len - user desc and news
#         # topic as retrieval
#
#         """history"""
#         if str(user_id) not in history_data:
#             history_text = ""
#             # continue
#         else:
#             history_posts = [item["text"] for item in history_data[str(user_id)][:50]]  # if int(item["tweet_id"]) != int(sample["tweet_id"])
#             history_text = ";".join(history_posts)
#
#         """check"""
#         if not user_desc and not history_text:
#             continue
#
#         if "predicted" not in sample or "predicted_intensity" not in sample:
#             continue
#
#         # filter out inactive users
#         post_text = post_data[str(post_id)]['text']
#         post_text = preprocess_tweet_local(post_text)
#
#         tgt_text = sample["text"]
#         # # skip the ones with url
#         # if p.clean(tgt_text)!=tgt_text:
#         #     continue
#         if find_URLS(tgt_text):
#             continue
#         tgt_text = p.clean(tgt_text)
#         if not tgt_text.strip():
#             continue
#         res_new.append({
#             "user_id": user_id,
#             "post_id": post_id,
#             "label_intensity": sample["predicted_intensity"],
#             "label_polarity": sample["predicted"],
#         })
#     return res_new


def get_graph_dataset(users_followings_dict, id_to_index):
    valid_uids_with_influencers = set(list(id_to_index))
    user_user_pairs = []
    for uid, followings in users_followings_dict.items():
        if uid not in valid_uids_with_influencers: continue
        uid_cur = uid
        for following in followings:
            following = str(following)
            if following not in valid_uids_with_influencers: continue
            following_cur = following
            if uid in valid_uids_with_influencers and following in valid_uids_with_influencers:  # strorint
                user_user_pairs.append([uid_cur, following_cur])

    return user_user_pairs


def get_text_embeddings(list_of_sentences, embedding_plm, device="cuda", batch_size=32, cache_path=None, use_cache=True):
    # Instruction: get embeddings for the first token of each sentence. args.plm is name of the pretrained language model.  list_of_sentences is a list of strings
    # Output: a numpy array of shape (len(list_of_sentences), plm_hidden_dim for the plm)

    cache_path = f"{CNN_dir}cache/{cache_path}"

    if path_exists(cache_path) and use_cache:
        print("loaded from ", cache_path)
        res_mat = torch.tensor(np.load(cache_path), dtype=torch.float)
        res_mat = res_mat.to("cpu")
        return res_mat

    plm = AutoModel.from_pretrained(embedding_plm)
    tokenizer = AutoTokenizer.from_pretrained(embedding_plm)
    plm.eval()
    plm.to(device)
    # args.batch_size is the batch size for the language model

    embeddings = []
    for i in tqdm(range(0, len(list_of_sentences), batch_size)):
        batch = list_of_sentences[i:i + batch_size]
        batch = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        batch = batch.to(device)
        with torch.no_grad():
            outputs = plm(**batch)
            embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
    embeddings = np.concatenate(embeddings, axis=0)
    # cache
    if cache_path is not None:
        np.save(cache_path, embeddings)
    # free up memory
    plm.cpu()
    del plm
    res_mat = torch.tensor(embeddings, dtype=torch.float)
    res_mat = res_mat.to("cpu")
    return res_mat


def get_top_k_text(query: str, list_of_sentences: List[str], embedding_plm, k: int, device="cuda", batch_size=32):
    """
    Get the top k sentences most similar to the query from the list_of_sentences using FAISS.

    Parameters:
    - query (str): The query sentence.
    - list_of_sentences (List[str]): A list of sentences.
    - embedding_plm (str): The pretrained language model for embeddings.
    - k (int): The number of top sentences to retrieve.
    - device (str, optional): The device type for PyTorch. Defaults to "cuda".
    - batch_size (int, optional): Batch size for processing sentences. Defaults to 32.

    Returns:
    List[str]: Top k sentences most similar to the query.
    """

    # Load model and tokenizer
    plm = AutoModel.from_pretrained(embedding_plm).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(embedding_plm)

    # Get embeddings
    all_sentences = [query] + list_of_sentences
    embeddings = []
    for i in tqdm(range(0, len(all_sentences), batch_size)):
        batch = all_sentences[i:i + batch_size]
        batch = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = plm(**batch)
            embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
    embeddings = np.concatenate(embeddings, axis=0).astype('float32')

    # Setup FAISS index
    d = embeddings.shape[1]  # dimension
    index = faiss.IndexFlatL2(d)  # build the index, L2 distance metric
    index.add(embeddings[1:])  # add vectors to the index, excluding query

    # Query FAISS index
    _, top_k_indices = index.search(embeddings[0].reshape(1, -1), k)  # search
    top_k_indices = top_k_indices.flatten()

    # Get top k sentences
    top_k_sentences = [list_of_sentences[index] for index in top_k_indices]

    return top_k_sentences


# # Assume get_text_embeddings is defined above as in your previous code
#
# def get_top_k_text(query: str, list_of_sentences: List[str], embedding_plm, k: int, device="cuda", batch_size=32):
#     """
#     Get the top k sentences most similar to the query from the list_of_sentences.
#
#     Parameters:
#     - query (str): The query sentence.
#     - list_of_sentences (List[str]): A list of sentences.
#     - embedding_plm (str): The pretrained language model for embeddings.
#     - k (int): The number of top sentences to retrieve.
#     - device (str, optional): The device type for PyTorch. Defaults to "cuda".
#     - batch_size (int, optional): Batch size for processing sentences. Defaults to 32.
#
#     Returns:
#     List[str]: Top k sentences most similar to the query.
#     """
#
#     # Concatenate the query to the list of sentences and get embeddings
#     all_sentences = [query] + list_of_sentences
#     all_embeddings = get_text_embeddings(all_sentences, embedding_plm, device, batch_size)
#
#     # Separate the query embedding from the rest
#     query_embedding = all_embeddings[0].numpy()
#     sentence_embeddings = all_embeddings[1:].numpy()
#
#     # Compute cosine similarities between query and all sentences
#     similarities = []
#     for sentence_embedding in sentence_embeddings:
#         similarity = 1 - cosine(query_embedding, sentence_embedding)
#         similarities.append(similarity)
#
#     # Get indices of top k similar sentences
#     top_k_indices = np.argsort(similarities)[-k:][::-1]
#
#     # Retrieve the top k sentences
#     top_k_sentences = [list_of_sentences[index] for index in top_k_indices]
#
#     return top_k_sentences


def reduce_pos_tag(s):
    if 'NN' in s:
        return 'NN'
    elif 'VB' in s:
        return 'VB'
    elif 'JJ' in s:
        return 'JJ'


@dataclass
class CustomCollatorRET():
    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    has_concepts: bool = False

    def __call__(self, features):
        # features=deepcopy(feats)

        encoder_features = [{'input_ids': feat['input_ids'],
                             'attention_mask': feat['attention_mask'],
                             } for feat in features]
        input_features = self.tokenizer.pad(
            encoder_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        input_features.update({
            'sample_ids': [feat['sample_id'] for feat in features]
        })

        return input_features


class MyDataset(Dataset):

    def __init__(self, data):
        super().__init__()
        """========Init========="""
        self.instances = data

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        instance = self.instances[idx]
        return instance


def gpu_setup(use_gpu=True, gpu_id="0"):  # , use_random_available=True
    print("\nSetting up GPU")
    # if len(gpu_id):
    #     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #     os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # print("visibes",os.environ["CUDA_VISIBLE_DEVICES"])
    num_gpus = 1
    if torch.cuda.is_available() and use_gpu:
        print(f"{torch.cuda.device_count()} GPU available")
        # print('cuda available with GPU:', torch.cuda.get_device_name(0))

        # use all
        device = torch.device("cuda")

    else:
        if use_gpu and not torch.cuda.is_available():
            print('cuda not available')
        device = torch.device("cpu")

    print("Device is set to", device)
    return device


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_word_embedding_matrix(words, model, tokenizer, prep_batch_size=40, args=None, cache_path=None, is_sbert=False):
    """
    Get the word embedding matrix for a list of words.
    """
    # Get the embedding matrix for the words in the list.
    device = gpu_setup(use_gpu=args.use_gpu, gpu_id=args.gpu_id)
    if path_exists(cache_path) and args.save_mat:
        print("loaded from ", cache_path)
        res_mat = torch.tensor(np.load(cache_path), dtype=torch.float)
        res_mat = res_mat.to("cpu")
        return res_mat

    max_seq_len = tokenizer.model_max_length if tokenizer.model_max_length <= 100000 else 512  # TODO
    print("max_seq_len", max_seq_len)
    batch_tokens = tokenizer(words, max_length=max_seq_len, truncation=True, padding=True)  # , return_tensors="pt" padding=True,
    inputs = [{"sample_id": j,
               "input_ids": batch_tokens["input_ids"][j],
               "attention_mask": batch_tokens["attention_mask"][j]}
              for j in range(len(batch_tokens["input_ids"]))]

    data_collator = CustomCollatorRET(tokenizer, model=model, max_length=max_seq_len, has_concepts=False)
    ds_dataloader = DataLoader(MyDataset(inputs), batch_size=args.prep_batch_size, shuffle=False, collate_fn=data_collator, drop_last=False)

    # embedding_matrix=model(**batch_tokens).last_hidden_state[:, 0, :]
    model = model.to(device)
    model.eval()

    logits_lists = []
    for batch in tqdm(ds_dataloader):
        for k, v in batch.items():
            if k != "sample_ids" and v is not None:
                batch[k] = v.to(device)  # cuda()

        with torch.no_grad():
            logits = model(input_ids=batch[f"input_ids"],
                           attention_mask=batch[f"attention_mask"],
                           return_dict=True)
            # logits_lists.extend(logits.tolist())
        if not is_sbert:
            logits = logits.last_hidden_state[:, 0, :]
            # Perform pooling
        if is_sbert:
            logits = mean_pooling(logits, batch['attention_mask'])

            # Normalize embeddings
            logits = F.normalize(logits, p=2, dim=1)

        logits_lists.append(logits.cpu())

    embedding_matrix = torch.cat(logits_lists, dim=0)

    np.save(cache_path, embedding_matrix.numpy())

    model = model.cpu()

    return embedding_matrix.to("cpu")


def get_sim_neighbours(a, b, cache_path, use_cache=True, valid_uids=None):
    # a/b is tuples of (user id, list of neighbor ids)
    # return a similarity matrix of size (len(a), len(b))
    # similarity is measured by the number of common neighbors
    cache_path = f"{CNN_dir}cache/{cache_path}"
    if use_cache and path_exists(cache_path):
        print("loaded from ", cache_path)
        res = torch.load(cache_path)
        return res

    res = torch.zeros(len(a), len(b))
    for i, (u1, n1) in enumerate(tqdm(a)):
        for j, (u2, n2) in enumerate(tqdm(b)):
            # res[i, j] = len(set(n1).intersection(set(n2)))
            l2 = len(n1.union(n2))
            res[i, j] = len(n1.intersection(n2)) / l2 if l2 > 0 else 0

    return res


def get_consine_sim_matrix(a, b, cache_path="post_profile.pt", use_cache=True):
    cache_path = f"{CNN_dir}cache/{cache_path}"
    if use_cache and path_exists(cache_path):
        print("loaded from ", cache_path)
        res = torch.load(cache_path)
        return res
    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    print("e3")

    res = torch.mm(a_norm, b_norm.transpose(0, 1))
    print("e4")
    return res


# def get_consine_sim_matrix2(sents1, sents2):
#     a_norm = a / a.norm(dim=1)[:, None]
#     b_norm = b / b.norm(dim=1)[:, None]
#     res = torch.mm(a_norm, b_norm.transpose(0, 1))
#     # print(res)
#     return res


def get_candidate_pool(word, record):
    pool = []
    lemmatizer = WordNetLemmatizer()
    lemm = lemmatizer.lemmatize(word)
    if word in record:
        return record[word]
    for item in wn.synsets(word, pos=wn.NOUN)[:1]:  # [:1]
        if lemm in item.name():
            for p in item.hypernyms():
                for c in p.hyponyms():
                    # print(c, p)
                    pool.extend([str(lemma.name()) for lemma in c.lemmas()])
    record[word] = pool
    return record[word]


def get_candidate_pool2(word, record, return_random=False, random_size=3):
    # assume word is a lemma
    pool = []
    # lemmatizer = WordNetLemmatizer()
    # lemm=lemmatizer.lemmatize(word)
    if word in record:
        return record[word] if not return_random else np.random.choice(record[word], size=min(len(record[word]), random_size), replace=False)
    for item in wn.synsets(word, pos=wn.NOUN)[:1]:  # [:1]
        if word in item.name():
            for c in item.hyponyms():
                # print(c, p)
                pool.extend([str(lemma.name()).replace("_", " ") for lemma in c.lemmas()])
            for c in item.hypernyms():
                # print(c, p)
                pool.extend([str(lemma.name()).replace("_", " ") for lemma in c.lemmas()])

    record[word] = pool
    return record[word] if not return_random else np.random.choice(record[word], size=min(len(record[word]), random_size), replace=False)


def compute_token2nodepos_batch(cur_ranges, bsz, seqlen, accumulate=False):
    token2nodeid = -torch.ones(bsz, seqlen, dtype=torch.long)
    start_pos = 0
    for batch_id, batch_range in enumerate(cur_ranges):
        for node_id, (s, e) in enumerate(batch_range):
            token2nodeid[batch_id, s:e] = node_id if not accumulate else start_pos + node_id
        start_pos += len(batch_range)
    return token2nodeid.long()


def compute_token2nodepos(cur_ranges, seqlen, pad_mask=None):
    token2nodeid = -torch.ones(seqlen, dtype=torch.long)
    for node_id, (s, e) in enumerate(cur_ranges):
        if pad_mask is not None and pad_mask[node_id] == 0:  # padding for entities
            continue
        token2nodeid[s:e] = node_id
    return token2nodeid.long()


def flatten_list(ls_batch):
    return [item for sublist in ls_batch for item in sublist]


def token_lens_to_idxs(batch_list_of_ranges):
    """Map token lengths to a word piece index matrix (for torch.gather) and a
    mask tensor.
    For example (only show a sequence instead of a batch):

    token lengths: [1,1,1,3,1]
    =>
    indices: [[0,0,0], [1,0,0], [2,0,0], [3,4,5], [6,0,0]]
    masks: [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.33, 0.33, 0.33], [1.0, 0.0, 0.0]]

    Next, we use torch.gather() to select vectors of word pieces for each token,
    and average them as follows (incomplete code):

    :param token_lens (list): token lengths.
    :return: a index matrix and a mask tensor.
    """

    # input is b x node (uneven)
    res = []

    max_token_num = max([len(b) for b in batch_list_of_ranges])
    max_token_len = max([(e - s) for b in batch_list_of_ranges for s, e in b])
    idxs, masks = [], []
    for b in batch_list_of_ranges:
        seq_idxs, seq_masks = [], []
        offset = 0
        for s, e in b:
            token_len = e - s
            seq_idxs.extend([i for i in range(s, e)]
                            + [0] * (max_token_len - token_len))  # -1
            seq_masks.extend([1.0 / token_len] * token_len
                             + [0.0] * (max_token_len - token_len))
        seq_idxs.extend([0] * max_token_len * (max_token_num - len(b)))
        seq_masks.extend([0.0] * max_token_len * (max_token_num - len(b)))
        idxs.append(seq_idxs)
        masks.append(seq_masks)
    # max_token_num = max([len(x) for x in token_lens])
    # max_token_len = max([max(x) for x in token_lens])
    # idxs, masks = [], []
    # for seq_token_lens in token_lens:
    #     seq_idxs, seq_masks = [], []
    #     offset = 0
    #     for token_len in seq_token_lens:
    #         seq_idxs.extend([i + offset for i in range(token_len)]
    #                         + [-1] * (max_token_len - token_len))
    #         seq_masks.extend([1.0 / token_len] * token_len
    #                          + [0.0] * (max_token_len - token_len))
    #         offset += token_len
    #     seq_idxs.extend([-1] * max_token_len * (max_token_num - len(seq_token_lens)))
    #     seq_masks.extend([0.0] * max_token_len * (max_token_num - len(seq_token_lens)))
    #     idxs.append(seq_idxs)
    #     masks.append(seq_masks)
    return idxs, masks, max_token_num, max_token_len


def merge_all_sents(sents, tokenizer=None, step_types_dict=None, step_types=None, has_cls=True):
    merged_sent = [tokenizer.cls_token] if has_cls else []
    start_pos = 1 if has_cls else 0  # cls existss
    start_positions = [start_pos]
    for i, sent in enumerate(sents):
        merged_sent += (sent + [step_types_dict[step_types[i]]])
        start_pos = len(merged_sent)
        start_positions.append(start_pos)
    return merged_sent, start_positions


def accumulate_ranges(ranges_batch, start_positions):
    for m, ranges in enumerate(ranges_batch):
        for i, (k, s, e) in enumerate(ranges):
            # res.append((s+start_positions[i], start_positions[i + 1]))
            ranges[i] = [s + start_positions[k], e + start_positions[k]]
    return ranges_batch


def update_ranges_with_subtoken_mapping(ranges_batch, new_map, max_seq_length, filter_oor=False):
    valid_range_mask = []
    res = []
    for i, ranges in enumerate(ranges_batch):
        tmp = []
        tmp2 = []
        for j, (s, e) in enumerate(ranges):
            if e not in new_map:
                breakpoint()
                embed()
            if new_map[e] < max_seq_length - 1:  # node_cnt == max_node_num
                tmp.append([new_map[s], new_map[e]])
                tmp2.append(1)
            else:
                if not filter_oor:
                    tmp.append([-1, -1])
                tmp2.append(0)
        if not filter_oor or len(tmp) > 1:
            res.append(tmp)
        valid_range_mask.append(tmp2)
    return res, valid_range_mask


def exclude_columns_for_lists(ls_batch, cols):
    for i, ls in enumerate(ls_batch):
        for j, l in enumerate(ls):
            ls[j] = [x for k, x in enumerate(l) if k not in cols]
    return ls_batch


# def aggregate_graphs(graphs, node_span_batch, valid_node_masks, max_node_num):
#     """
#
#     :param graphs: list of graphs, [[[0,1], [1,0]]
#     :example:
#     :param valid_node_masks: [[-1,1,1,-1]]
#     :param max_node_num:
#     :return:
#     """
#     """====Convert node ranges to new token indices===="""
#     graph_instances = []
#     node_cnt = 0
#
#     for i, (n_mask, graph) in enumerate(zip(valid_node_masks, graphs)):
#
#         node_spans = node_span_batch[i]
#         edge_index = np.array(graph['edge_index'])
#         edge_attr = graph['edge_attr']
#
#         num_valid_nodes = sum(n_mask)
#         # num_invalid_nodes=len(n_mask) - num_valid_nodes
#         # e_mask = [0 if (n_mask[s] == 0 or n_mask[e] == 0) else 1 for s, e in edge_index]
#         # num_valid_edges = sum(e_mask)
#         # num_invalid_edges = len(e_mask) - num_valid_edges
#         if num_valid_nodes + node_cnt > max_node_num:
#             break
#
#         if num_valid_nodes == len(n_mask):
#             # graph_instances.append(graph)
#             # root_indices.append(node_cnt)
#             node_cnt += len(n_mask)  # num_valid_nodes
#             graph_instances.append(Data(x=get_tensor_long(node_spans),
#                                         edge_index=get_tensor_long(edge_index.T),
#                                         edge_attr=get_tensor_long(edge_attr)))
#
#     if not graph_instances:
#         graph_instances.append(Data(x=get_tensor_long([]), edge_index=get_tensor_long([]), edge_attr=get_tensor_long([])))
#         print(f"empty graph")
#         embed()
#     return Batch.from_data_list(graph_instances)  # , root_indices


def sents_to_token_ids_accumulate(sents=None, max_seq_length=None, tokenizer=None,
                                  special_tks=None, step_types_dict=None, use_special_tag=False, list_of_ranges=None, list_of_ent_ranges=None, step_types=None, max_node_num=200):
    # print("\n\nsents_to_token_ids_with_graph")
    """====update word indices after merging all steps===="""
    # special_tks = ["[GOAL]", "[SUBGOAL]", "[STEP]", tokenizer.sep_token, "<ROOT>"]
    #
    # # g is like x, edge index, edge attr
    # step_types_dict = {
    #     "goal": "[GOAL]",
    #     "subgoal": "[SUBGOAL]",
    #     "event": "[STEP]",
    # }
    #
    # if not use_special_tag:
    #     step_types_dict = {k: tokenizer.sep_token for k, v in step_types_dict.items()}

    merged_sent = [tokenizer.cls_token]
    start_pos = 1  # cls existss
    for i, ent2ranges in enumerate(list_of_ent_ranges):
        # ranges=
        for ent, ranges in ent2ranges.items():
            if len(ranges):  # empty graph
                # assert len(ent2ranges[ent])==2
                # if -1 in ent2ranges[ent][1]: ent2ranges[ent].pop()
                ent2ranges[ent] = (np.array(ranges) + start_pos).tolist()

        # if len(ranges): # empty graph
        #     list_of_ranges[i]=(np.array(ranges) + start_pos).tolist()
        merged_sent += (sents[i] + [step_types_dict[step_types[i]]])
        start_pos = len(merged_sent)

    input_ids, new_map, _, _, attention_mask = sent_to_token_ids(merged_sent, max_seq_length, tokenizer, shift_right=False, add_sep_at_end=True, has_cls_at_start=True,
                                                                 special_tks=special_tks)

    out_of_range = False
    node_cnt = 0

    for i, ent2ranges in enumerate(list_of_ent_ranges):
        # ranges=
        # already out of range
        if out_of_range:
            list_of_ent_ranges[i] = {}
            continue
        for ent, ranges in list(ent2ranges.items()):
            for j, (s, e) in enumerate(ranges):
                if e not in new_map:
                    embed()
                if new_map[e] >= max_seq_length - 1 or node_cnt == max_node_num:
                    out_of_range = True
                    ent2ranges.pop(ent)
                    break
                ent2ranges[ent][j][0] = new_map[s]
                ent2ranges[ent][j][1] = new_map[e]
                node_cnt += 1
    return input_ids, attention_mask, list_of_ent_ranges  # list_of_ranges


def sent_to_token_ids(sent, max_seq_length, tokenizer, shift_right=False, add_sep_at_end=True, has_cls_at_start=True, special_tks=None, end_token=None):
    """
    @param sent: list of tokens, with cls
    @param ent_pos_list: list of s e index pairs for each mention, like [[0,1],[5,7]]
    @param max_seq_length: max bert seqlen
    @param tokenizer: tokenizer
    @param shift_right: always set true, shift new mention position in tokens +1 because we have CLS
    @param add_marker: add * to before and after mention
    @return: list of tokens, and updated ent_pos_list
assume no sep
    """

    new_map = {}
    original_pos_vec = []
    sents = []

    if shift_right:
        front_token = tokenizer.cls_token
        if front_token is None: front_token = tokenizer.bos_token
        if front_token is None: front_token = tokenizer.pad_token  # T5
        sents = [front_token] + sent

    for i_t, token in enumerate(sent):
        token = token.strip()
        if not len(token):
            token = " "  # prevent empty token to disappear, making

        # tokens_wordpiece = tokenizer.tokenize(token)
        after_the_first_token = (has_cls_at_start and i_t > 1) or (not has_cls_at_start and i_t > 0)
        if token in special_tks or (after_the_first_token and sent[i_t - 1].strip() in special_tks):  # for gpt like tokenizer which cares about space
            tokens_wordpiece = tokenizer.tokenize(token)
        else:
            tokens_wordpiece = tokenizer.tokenize((" " + token) if after_the_first_token else token)

        # if tokenizer.sep_token == token or (i_t > 1 and sent[
        #     i_t - 1].strip() == tokenizer.sep_token):  # for gpt like tokenizer which cares about space
        #     tokens_wordpiece = tokenizer.tokenize(token)
        # else:
        #     tokens_wordpiece = tokenizer.tokenize(
        #         " " + token if ((has_cls_at_start and i_t > 1) or (not has_cls_at_start and i_t > 0)) else token)
        new_map[i_t] = len(sents)
        for _ in range(len(tokens_wordpiece)):
            original_pos_vec.append(i_t)
        sents.extend(tokens_wordpiece)
    new_map[i_t + 1] = len(sents)

    # sents = sents[:max_seq_length - 2]
    if has_cls_at_start:
        sents = sents[:max_seq_length - 1]
    else:
        sents = sents[:max_seq_length - 2]
    if add_sep_at_end:
        # end_token = tokenizer.sep_token
        if end_token is None: end_token = tokenizer.eos_token

        sents += [end_token]
        # new_map[i_t + 2] = len(sents) # no need since sep will be moved left is sent too long

        original_pos_vec.append(original_pos_vec[-1] + 1)

    input_ids = tokenizer.convert_tokens_to_ids(sents)
    # print("sents",sents)
    token_type_ids = [0] * len(input_ids)
    attention_mask = [1] * len(input_ids)
    return input_ids, new_map, original_pos_vec, token_type_ids, attention_mask, sents


def modify_output(s, tokenizer, is_tgt=False):
    tmp = s.strip()
    if tokenizer.cls_token is not None:
        tmp = tmp.replace(tokenizer.cls_token, "")
    tmp = tmp.replace(tokenizer.pad_token, "")

    # if tokenizer.sep_token is not None:
    #     s, _ = tmp.split(tokenizer.sep_token)[0], "\n".join(tmp.split(tokenizer.sep_token)[1:])
    # else:
    #     s, _ = tmp.split(tokenizer.eos_token)[0], "\n".join(tmp.split(tokenizer.eos_token)[1:])

    if is_tgt:
        s = tmp.replace(". ", f"\n" + " " * 18)

    else:
        s = tmp.replace(tokenizer.sep_token, f" {tokenizer.sep_token}\n" + " " * 18)
        if '[GOAL]' in s:
            s = s.replace("[GOAL]", " [GOAL]\n" + " " * 18). \
                replace("[SUBGOAL]", " [SUBGOAL]\n" + " " * 18). \
                replace("[STEP]", " [STEP]\n" + " " * 18)
    return s.strip()


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def clear_subgoals3(step_ids, step_db):
    steps = [step_db[id] for id in step_ids]
    step_types = [step["step_type"] for step in steps]
    step_ids_new = []
    for i, step_id in enumerate(step_ids):
        if step_types[i] == "subgoal":
            continue
        step_ids_new.append(step_id)
    return step_ids_new


def clear_subgoals2(steps):
    steps_new = []
    step_types_new = []
    for i, step in enumerate(steps):
        if step["step_type"] == "subgoal":
            continue
        steps_new.append(step)
    return steps_new


def clear_subgoals(steps, step_types):
    steps_new = []
    step_types_new = []
    for i, step in enumerate(steps):
        if step_types[i] == "subgoal":
            continue
        steps_new.append(step)
        step_types_new.append(step_types[i])
    return steps_new, step_types_new
