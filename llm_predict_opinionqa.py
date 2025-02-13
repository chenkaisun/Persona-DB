# import chunk
import random
import numpy as np
import requests
import os
import json
from tqdm import tqdm
import math
from utils.utils import *
from constants import *
from collections import defaultdict, Counter
# import networkx as nx
# import preprocessor as p
from llm_api import chatbot
# from statsmodels.stats.inter_rater import fleiss_kappa
from copy import deepcopy
from transformers import GPT2Tokenizer
# from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, mean_squared_error, mean_absolute_error, \
#     precision_score, recall_score, cohen_kappa_score
# from scipy.stats import pearsonr, spearmanr
from evaluate import eval_tuple_response
from utils.data_utils import get_user_profile2, retrieve_api_embedding, UserRetriever, standardize_keys, process_triplet, load_or_create_dataframe, update_dataframe, remove_dup
from datetime import datetime
import wandb
from sklearn.metrics import precision_recall_fscore_support
if module_exists("llmlingua"):
    from llmlingua import PromptCompressor


# args
parser = argparse.ArgumentParser()
parser.add_argument("--num_nodes", type=int, default=1)
parser.add_argument("--node_id", type=int, default=0)
parser.add_argument("--num_times", type=int, default=1)
parser.add_argument("--case_study", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--pure_intensity", type=int, default=0)
parser.add_argument("--use_refined_profile", type=int, default=0)
parser.add_argument("--is_eval", type=int, default=1)
parser.add_argument("--dataset", type=str, default="34")
parser.add_argument("--debug", type=int, default=0)
parser.add_argument("--experiment", type=str, default="")
parser.add_argument("--neighbor_threshold", type=float, default=0) #0.6
parser.add_argument("--item_threshold", type=float, default=0)
parser.add_argument("--neighbor_topk", type=int, default=3)
parser.add_argument("--item_topk", type=int, default=30)
parser.add_argument("--item_topk_collab", type=int, default=10)
parser.add_argument('--prompt_types', default=["prediction"], nargs='+')
parser.add_argument('--cases', default=["", ], nargs='+')
parser.add_argument('--n_hist', type=int, default=100)
parser.add_argument('--include_hist', type=int, default=1)
parser.add_argument('--include_empty_neighbors', type=int, default=0)
parser.add_argument('--do_process', type=int, default=0)
parser.add_argument('--model', type=str, default="gpt-3.5-turbo-0613") # "gpt-4-1106-preview"
parser.add_argument('--gen_eval', type=int, default=0) # "gpt-4-1106-preview"
parser.add_argument('--modules', default=["hist","r","rf"], nargs='+')
parser.add_argument('--nb_hist_only', default=1, type=int)
parser.add_argument('--longer_ver', default=1, type=int)
parser.add_argument('--include_rf', default=0, type=int) #1 rf 2 r 3 both
parser.add_argument('--lc_model', default="xlm", type=str) # lingua


args = parser.parse_args()
"""auto args.modules"""
if args.include_rf:
    args.modules = ["hist", "rf"]
"""always args.do_process 0 since refined_history_vec=remove_dup(refined_history_vec)"""
args.do_process=0


CNN_dir = f"{data_dir}OpinionQA/human_resp/American_Trends_Panel_W{args.dataset}/"
CNN_test_anno_filename = f"{CNN_dir}response_data_balanced_test_anno.json"

CNN_post_filename = f"{CNN_dir}post_data.json"
CNN_users_info_rf_filename = f"{CNN_dir}users_info_dict_with_chatgptanno_rf.json"
CNN_users_history_filename = f"{CNN_dir}users_history_dict.json"
CNN_prediction_dir=f"{CNN_dir}predictions/"
CNN_results_dir=f"{CNN_dir}results/"
CNN_prediction_prompt_dir=f"{CNN_dir}prediction_prompts/"
CNN_api_embeddings_dir=f"{CNN_dir}api_embeddings/"
CNN_r_anno_json_dir=f""
CNN_rf_anno_json_dir=f""
CNN_non_empty_hist_uids_filename=f""
CNN_non_empty_hist_uids_processed_filename=f""

if args.longer_ver:
    CNN_test_anno_filename = f"{CNN_dir}response_data_balanced_test_anno_1000.json"
    CNN_prediction_dir=f"{CNN_dir}predictions_1k/"
    CNN_results_dir=f"{CNN_dir}results_1k/"
    CNN_prediction_prompt_dir=f"{CNN_dir}prediction_prompts_1k/"

print(args)
print(CNN_dir)

# parser.add_argument('--eval_pi', type=int, default=0)

# """change CNN_dir in constant for different datasets"""
# parse
wandb.init()

# if args.debug and not args.experiment:
#     args.experiment="debug"

print("args.cases", args.cases)
print("args.modules", args.modules)
random.seed(0)
np.random.seed(0)

model = "text-embedding-ada-002"

"""==========================================Metadata=============================================="""
# train, dev, test = load_file_batch([CNN_train_filename, CNN_dev_filename, CNN_test_filename])
train_anno, dev_anno, test_anno = load_file_batch([CNN_test_anno_filename, CNN_test_anno_filename, CNN_test_anno_filename])
train, dev, test = train_anno, dev_anno, test_anno
post_data, user_data = load_file_batch(filenames=[CNN_post_filename, CNN_users_info_rf_filename])
# users_followings_dict = load_file(CNN_users_followings_dict_filename)
valid_uids_test={item["author_id"] for item in test_anno}

history_data = load_file(CNN_users_history_filename)
# meta data above
gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

"""==========================================Get pred=============================================="""

def calculate_metrics(y_true_list, y_pred_list):
    precisions, recalls, f1_scores = [], [], []
    for y_true, y_pred in zip(y_true_list, y_pred_list):
        precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)
    return precisions, recalls, f1_scores

def aggregate_metrics(metrics):
    precision_macro = sum(metrics[0]) / len(metrics[0])
    recall_macro = sum(metrics[1]) / len(metrics[1])
    f1_score_macro = sum(metrics[2]) / len(metrics[2])
    return precision_macro, recall_macro, f1_score_macro

def parse_output(output):
    try:
        lines = [item.strip() for item in output.strip().split('\n') if item.strip()]
        answer=""
        if len(lines)>0:
            answer = lines[0].split(': ')[1].strip()
        explanation=" "
        if len(lines)>1:
            explanation = lines[1].split(': ')[1].strip()
    except:
        print("output\n\n", output)
        # exit()
        answer, explanation="",""

    return answer, explanation

# Set variables
is_eval = args.is_eval
pure_intensity = args.pure_intensity
num_times = args.num_times
print("args.is_eval", args.is_eval)
new_user_data_sc = load_file(CNN_users_info_rf_filename)

prompt_types_tmp = args.prompt_types
prompt_types = []
for nm in range(1, num_times + 1):
    for tp in prompt_types_tmp:  # 'prediction',
        # if nm==0:
        #     prompt_types.append(f"{tp}")
        # else:
        prompt_types.append(f"{tp}_{nm}")

# Case Studies
case_study_dict = {}
if path_exists(f"{CNN_dir}case_study_dict.json"):
    case_study_dict = load_file(f"{CNN_dir}case_study_dict.json")
    case_study_dict = {k: set(v) for k, v in case_study_dict.items()}
case_study_keys = args.cases
print("\n\ncase_study_keys", case_study_keys)

if args.debug:
    case_study_keys = ['']
if args.case_study == 1:
    exit()
    test_anno = load_file(CNN_test_anno_lurker_filename)
    case_study_keys = ['']
if args.case_study == 2:
    case_study_keys = ['unseen_test']
if case_study_keys == ['longest_hist_100_all']:
    print("longest_hist_100_all")
    test_anno = train_anno + dev_anno + test_anno

is_eval_set=False
if case_study_keys!=[""] and all([not c.endswith("test") for c in case_study_keys]):
    print("[not c.endswith(test) for c in case_study_keys]")
    test_anno = dev_anno + test_anno
    is_eval_set=True

cur_valid_uids={item["author_id"] for item in test_anno}
for case in case_study_keys:
    if case.strip():
        assert case_study_dict[case] & cur_valid_uids == case_study_dict[case]

# assert all tweetids are uniqe
# tweet_ids = [item["tweet_id"] for item in test_anno]
# assert len(tweet_ids) == len(set(tweet_ids))

# Eval Template
result_dict = load_file(f"../INCAS/analysis/results_chatgpt.json") if path_exists(f"../INCAS/analysis/results_chatgpt.json") else {}  # pd.DataFrame()
eval_keys = ["spearman", "pearson", "mif1", 'maf1', 'accuracy', 'mse','mae','wasserstein','Bleu_1', 'Bleu_2', "Bleu_3", "Bleu_4", 'meteor', 'rouge1', 'rouge2', 'rougeL', "avg_prompt_tokens", "avg_hist_tokens","pi_prompt", "pi_hist", "pi_prompt2", "pi_hist2"]

# eval_keys = ["spearman", "pearson", "mif1", 'maf1']

def make_llm_prediction(history, post_text, profile, prompt_type, use_refined_profile=1, augment=None, collab_history=None, choices=None,args=args):
    new_user_data = new_user_data_sc
    if not use_refined_profile:
        cur_key = parse_gpt_anno(new_user_data[str(user_id)]["chatgptanno"]) if str(user_id) in new_user_data and "chatgptanno" in new_user_data[str(user_id)] else ""
    else:
        cur_key = new_user_data[str(user_id)]["chatgptanno_r"]  # parse_gpt_anno(new_user_data[str(user_id)]["chatgptanno_r"])
        # if str(user_id) in new_user_data and "chatgptanno_r" in new_user_data[str(user_id)] else ""
    if augment is not None:  # mean history is mixture of analysis and posts and key is strcuturied
        cur_key = augment
    cur_sc = parse_gpt_anno(new_user_data_sc[str(user_id)]["sc"]) if str(user_id) in new_user_data_sc and "sc" in new_user_data_sc[str(user_id)] else ""
    cur_refine = new_user_data_sc[str(user_id)]["chatgptanno_rf"] if "chatgptanno_rf" in new_user_data_sc[str(user_id)] else ""
    # parse_gpt_anno(new_user_data_sc[str(user_id)]["refine"]) if str(user_id) in new_user_data_sc and "refine" in new_user_data_sc[str(user_id)] else ""

    # if not cur_sc.strip() and not case.strip():
    #     print("no sc", user_id)
    #     continue

    load_from_orig = False

    prev_file = f"{CNN_prediction_dir}chatgpt_pred_{prompt_type}_{version.split('_')[0]}/{tweet_id}.txt" if args.case_study == 1 else ""
    if args.case_study == 1 and path_exists(prev_file):
        pred_fname = prev_file
    elif not cur_sc.strip() and 'prediction_with_prompt_sc' in prompt_type:
        load_from_orig = True
        pred_fname = f"{CNN_prediction_dir}chatgpt_pred_prediction_with_prompt_{version}/{tweet_id}.txt"
        if args.case_study == 1 and not path_exists(pred_fname):
            load_from_orig = False
            pred_fname = f"{CNN_prediction_dir}chatgpt_pred_{prompt_type}_{version}/{tweet_id}.txt"
    else:
        pred_fname = f"{args.prediction_dir}{tweet_id}.txt"
        #
        # f"{CNN_prediction_dir}chatgpt_pred_{prompt_type}_{version}{'_' + args.experiment if args.experiment else ''}/{tweet_id}.txt"


    cur_prompt_prediction = get_prompt(history=history, post=post_text, profile=profile,
                                       key=cur_key, nbs=None, sc=cur_sc, prompt_type=prompt_type, refine=cur_refine, neighbors=collab_history, choices=choices, prompts_dir="prompts/OpinionQA/")
    if not path_exists(pred_fname) and not load_from_orig:
        # print(f"not exist {user_id} {post_id} {tweet_id}")
        # breakpoint()
        print("current", user_id)
        # if user_id=="3091694828": breakpoint()
        dump_file(cur_prompt_prediction, f"{args.prediction_prompt_dir}{tweet_id}.txt")

        res = chatbot(cur_prompt_prediction, seed=args.seed, model=args.model)
        dump_file(res, pred_fname)

        # print("\n", res)
        print("finished", user_id)
    return pred_fname, cur_prompt_prediction


def get_avg_result(is_eval, result_dict, eval_keys):
    if not is_eval:
        return
    for case in result_dict:
        for prompt_type in result_dict[case]:
            for keyname in eval_keys:
                tmp = []
                for version in result_dict[case][prompt_type]:
                    if version != "average" and keyname in result_dict[case][prompt_type][version]:
                        tmp.append(result_dict[case][prompt_type][version][keyname])
                # Compute average
                if len(tmp) >= 3:
                    result_dict[case][prompt_type].setdefault("average", {})
                    result_dict[case][prompt_type]["average"][keyname] = round(np.mean(tmp), 2)
            # print("avg", case, prompt_type, result_dict[case][prompt_type]["average"])
    dump_file(result_dict, f"{CNN_results_dir}results_chatgpt.json")  # ../INCAS/analysis/
    print("result_dict", result_dict)


ur = UserRetriever(args=args, user_data=history_data, user_desc=user_data)
run_name="retrieval" if not args.do_process else "retrieval_dp"
for j, prompt_type in enumerate(prompt_types):
    # split last _
    """=================Init Suffix=================="""
    prompt_type, version = prompt_type.rsplit('_', 1)
    if args.case_study == 1:
        version = f"{version}_lurker"  # here lurker might draw from training data so it's an independent set

    print(f"prompt_type: {prompt_type}")
    print(f"version: {version}")

    """=================Init Eval Vectors=================="""
    sent_preds = []
    sent_labels = []

    intensity_preds = []
    intensity_labels = []

    decoded_preds = []
    decoded_labels_special = []

    num_tokens_prompt = []
    num_tokens_hist = []

    test_anno_batches = batchify(test_anno, math.ceil(len(test_anno) / args.num_nodes))
    if int(args.node_id) >= len(test_anno_batches):
        print("node_id is larger than the number of batches")
        continue
    cur_test_anno = test_anno_batches[args.node_id]

    """=================Loop Cases=================="""
    for case in case_study_keys:
        # case=args.case_study
        print("case is", case)
        if args.case_study not in result_dict:
            result_dict[args.case_study] = {}
        if prompt_type not in result_dict[args.case_study]:
            result_dict[args.case_study][prompt_type] = {}
        cnt = 0

        # if not is_eval_set:
        #     print("case set to '' since not is_eval_set")
        #     casename=""
        # now = datetime.now()
        # current_time = now.strftime("%Y%m%d%H%M%S")
        # base_name = f"chatgpt_case{case}_{prompt_type}_neighbor{args.neighbor_topk}_item{args.item_topk}_nh{args.n_hist}_ih{args.include_hist}_en{args.include_empty_neighbors}_dp{args.do_process}_pint{pure_intensity}_{version}"
        mdname=""
        if set(args.modules)!=set(["hist","r","rf"]):
            mdname = "md_"+"".join(args.modules)+"_"
        rname=""
        if args.include_rf:
            rname = f"if{args.include_rf}_"
        nhl = ""
        if args.include_rf:
            nhl = f"nhl{args.nb_hist_only}_"

        lc_type=""
        if args.lc_model!="xlm":
            lc_type=f"lc{args.lc_model}_"

        base_name = f"chatgpt_case{'' if not is_eval_set else 'eval_set'}_{prompt_type}_neighbor{args.neighbor_topk}_item{args.item_topk}_itemcb{args.item_topk_collab}_it{args.item_threshold}_nh{args.n_hist}_ih{args.include_hist}_en{args.include_empty_neighbors}_dp{args.do_process}_pint{pure_intensity}_{lc_type}{nhl}{rname}{mdname}{version}"

        if "_ret" not in prompt_type:
            base_name = f"chatgpt_case{'' if not is_eval_set else 'eval_set'}_{prompt_type}_neighbor{args.neighbor_topk}_item{args.item_topk}_itemcb{args.item_topk_collab}_it{args.item_threshold}_hl{2500}_nh{args.n_hist}_ih{args.include_hist}_en{args.include_empty_neighbors}_dp{args.do_process}_pint{pure_intensity}_{lc_type}{nhl}{rname}{mdname}{version}"

        # base_name = f"chatgpt_case{'' if not is_eval_set else 'eval_set'}_{prompt_type}_neighbor{args.neighbor_topk}_item{args.item_topk}_itemcb{args.item_topk_collab}_it{args.item_threshold}_nh{args.n_hist}_ih{args.include_hist}_en{args.include_empty_neighbors}_dp{args.do_process}_pint{pure_intensity}_md__{mdname}__{version}"
        # tmp_dir = f"{base_name}/" args.item_topk_collab, args.item_threshold,
        print("base_name", base_name)

        args.prediction_dir, args.prediction_prompt_dir = f"{CNN_prediction_dir}{base_name}/", f"{CNN_prediction_prompt_dir}{base_name}/"
        os.makedirs(args.prediction_dir, exist_ok=True)
        os.makedirs(args.prediction_prompt_dir, exist_ok=True)
        # our_dir = f"{CNN_prediction_dir}chatgpt_case{case}_{prompt_type}_neighbor{args.neighbor_topk}_item{args.item_topk}_pint{pure_intensity}_{version}/"
        # mkdir(our_dir)

        post_ids = []
        for i, sample in enumerate(tqdm(cur_test_anno)):  # this indicates ith path after breaking out all articles into individual paths
            sample["author_id"] = str(sample["author_id"])
            user_id = str(sample["author_id"])
            sample['tweet_id']=str(i)
            if case.strip() and user_id not in case_study_dict[case]:
                continue
            load_from_orig = False

            # cur_pred_fname = f"{CNN_prediction_dir}chatgpt_pred_{prompt_type}_{version}/{sample['tweet_id']}.txt"
            cur_pred_fname = f"{args.prediction_dir}{sample['tweet_id']}.txt"
            if path_exists(cur_pred_fname) and not args.is_eval:
                continue

            """=================Get Data Sample=================="""
            post_id = sample["conversation_id"]
            tweet_id = sample["tweet_id"]

            cur_top_n_hist = 100 if "_ret" in prompt_type else args.n_hist
            user_desc, history_text, history_posts, history_tweet_ids = get_user_profile2(user_id, user_data, history_data, top_n_hist=cur_top_n_hist,post_id=post_id)
            len_tokens = len(gpt_tokenizer.encode(history_text))

            tgt_text = "a"
            post_text = post_data[str(post_id)]['question']
            choices = post_data[str(post_id)]['options']
            choices_text="\n".join([f"{item}" for item in choices.keys() if int(choices[item])!=99])
            predicted = int(choices[sample['label'].lower().strip()])
            if predicted==99: continue
            predicted_intensity = predicted
            print("user_id", user_id)

            if is_eval and path_exists(cur_pred_fname):  # and not args.eval_pi
                res = load_file(cur_pred_fname)
                try:
                    sent_pred, explanation = parse_output(res)
                    sent_pred=sent_pred.lower().strip().strip("\'").strip("\"")
                    if sent_pred not in choices:
                        print("sent_pred", sent_pred)
                        print(f"error output sample {i}", res)
                    sent_pred=choices[sent_pred] if sent_pred in choices else 0
                    intensity_pred, decoded_pred = sent_pred, "a"
                except:
                    print("error filename", cur_pred_fname)
                    exit()

                # print(sent_pred, intensity_pred, decoded_pred)
                sent_preds.append(int(sent_pred))
                sent_labels.append(predicted)
                intensity_preds.append(int(intensity_pred))
                intensity_labels.append(abs(predicted_intensity - 3) if pure_intensity else predicted_intensity)
                decoded_preds.append(decoded_pred)
                decoded_labels_special.append([tgt_text])
                post_ids.append(post_id)

                if path_exists(args.prediction_prompt_dir + sample['tweet_id'] + ".txt"):
                    tmp_cur_prompt_prediction = load_file(args.prediction_prompt_dir + sample['tweet_id'] + ".txt")
                    num_tokens_prompt.append(len(gpt_tokenizer.encode(tmp_cur_prompt_prediction)))
                    hist_content_prompt = tmp_cur_prompt_prediction[tmp_cur_prompt_prediction.find("[user historical"):]
                    num_tokens_hist.append(len(gpt_tokenizer.encode(hist_content_prompt)))
                else:
                    tmp_cur_prompt_prediction=" "
                    num_tokens_prompt.append(10000)
                    num_tokens_hist.append(10000)

                continue

            len_tokens_limit = 2500
            if "hist" in prompt_type: 
                len_tokens_limit = 2500
            while len_tokens > len_tokens_limit and "_ret" not in prompt_type:
                print("too long", len_tokens)
                cur_top_n_hist -= 1 #
                user_desc, history_text, history_posts, history_tweet_ids = get_user_profile2(user_id, user_data, history_data, top_n_hist=cur_top_n_hist,post_id=post_id)

                len_tokens = len(gpt_tokenizer.encode(history_text))

            news, profile, history = post_text, user_desc, history_text

            if "_lc" in prompt_type:
                os.environ["TOKENIZERS_PARALLELISM"] = "false"

                if args.lc_model=="lingua":
                    llm_lingua = PromptCompressor("TheBloke/Llama-2-7b-Chat-GPTQ", model_config={"revision": "main"})
                    history = llm_lingua.compress_prompt(history, rate=0.40)['compressed_prompt']
                elif args.lc_model=="llingua":
                    llm_lingua = PromptCompressor("TheBloke/Llama-2-7b-Chat-GPTQ", model_config={"revision": "main"})
                    history = llm_lingua.compress_prompt(
                        [history],
                        question="Predict response from the user to a question from a Pew survey given the history data.",
                        rate=0.4,
                        # Set the special parameter for LongLLMLingua
                        condition_compare=True,
                        context_budget="+100",
                        rank_method="longllmlingua",
                    )['compressed_prompt'].split("\n")[:-1]
                    history="\n".join(history)

                elif args.lc_model=="xlm":
                    llm_lingua = PromptCompressor(
                        model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                        use_llmlingua2=True,  # Whether to use llmlingua-2
                    )
                    history = llm_lingua.compress_prompt(history, rate=0.40, force_tokens=['\n', '?'])['compressed_prompt']

            # cur_prompt_prediction = get_prompt(history=history, post=post_text, profile=profile, key=new_user_data["chatgptanno"], nbs=None, sc=new_user_data['sc'], prompt_type=prompt_type)

            """==================================Retrieve==================================="""
            struc_vec, augment = None, None
            collab_history = ""
            if "_ret" in prompt_type:

                # init struc string and history vec
                struc_vec = []
                history_tweet_ids = [f"{user_id}_orig_{item}" for item in history_tweet_ids]
                history_vec = list(zip(history_tweet_ids, history_posts))
                all_keys = desc_keys + list_keys

                # get history struc
                refined_history_vec = []
                # tmp = load_file(f"{CNN_dir}chatgptanno_r_json/{user_id}.json")
                # process_triplet(user_id, struc_vec, refined_history_vec, tmp, refine_type="r", do_process=args.do_process)
                # if "r" not in args.modules: # struc needs to be in
                #     refined_history_vec=[]
                # # print("r dict", tmp)
                #
                # get history refine
                if "rf" in args.modules and args.include_rf in [1,3]: # struc needs to be in
                    tmp = load_file(f"{CNN_dir}chatgptanno_rf_json/{user_id}.json", init={}) if not args.debug else {}
                    process_triplet(user_id, struc_vec, refined_history_vec, tmp, refine_type="rf", do_process=args.do_process)
                # print("rf dict", tmp)
                # if args.do_process: refined_history_vec=remove_dup(refined_history_vec)

                # augment = ""
                # if args.include_r or args.include_rf:
                #     if args.include_rf:
                #         tmp = load_file(f"{CNN_dir}chatgptanno_rf_json/{user_id}.json") if not args.debug else {}
                #         process_triplet(user_id, struc_vec, refined_history_vec, tmp, refine_type="rf", do_process=args.do_process)
                    # if args.do_process: refined_history_vec=remove_dup(refined_history_vec)
                augment = "\n".join(sorted(struc_vec))

                history_prompt = "Encode the user's historical survey question answer pair for retrieval. Even if not directly related to the current question, this data may still hint at the user's opinions and personality. Consider the data carefully. Data: "
                refined_history_prompt = "Encode the analysis on the user's historical survey answers for retrieval. Even if not directly related to the current question, this analysis on the user may still hint at the user's opinions and personality. Consider the analysis carefully. Analysis: "

                if prompt_type in ["prediction_ret"] or args.modules==["hist"]: # or args.modules==["hist"]
                    hvr = [(hist_id, history_prompt + hist_item) for hist_id, hist_item in history_vec]
                    combined_history_vec = history_vec
                else:
                    if args.include_hist:
                        hvr = ([(hist_id, history_prompt + hist_item) for hist_id, hist_item in history_vec] +
                               [(hist_id, refined_history_prompt + hist_item) for hist_id, hist_item in refined_history_vec])
                        combined_history_vec = history_vec + refined_history_vec
                    else:
                        hvr = [(hist_id, refined_history_prompt + hist_item) for hist_id, hist_item in refined_history_vec]
                        combined_history_vec = refined_history_vec

                query = "Encode this survey question to find the most relevant user past survey question answer pairs that might hint at their opinion to it: " + news
                if args.include_rf and ("rf" in args.modules or "r" in args.modules): # struc needs to be in
                    query = "Encode this survey question to find the most relevant user information that might hint at their opinion to it: " + news

                # print("News", news)

                # retrieve
                top_indices = retrieve_api_embedding((post_id, query), hvr, top_k=args.item_topk, run_name=run_name, save=True, threshold=0,dataset=args.dataset) #args.item_threshold
                history = "\n".join([combined_history_vec[i][1] for i in top_indices])
                # print("top posts")
                # print("\n".join([combined_history_vec[i][1] for i in top_indices]))

                """=================Get Neighbor Information=================="""

                if "collab" in prompt_type and ur.get_user_profile(user_id, user_desc).strip():
                    # print("e0")
                    retrieved_uids = ur.retrieve_users(user_id, top_k=args.neighbor_topk,
                                                       include_empty_neighbors=args.include_empty_neighbors)  # , threshold=args.neighbor_threshold
                    # print("e1")
                    refined_history_vec = ur.join_db(retrieved_uids, nb_hist_only=True)
                    if args.do_process: refined_history_vec=remove_dup(refined_history_vec)
                    # for hist_id, cur_item in refined_history_vec:
                    #     if not (cur_item.strip().lower().startswith("view toward") or cur_item.strip().lower().startswith("the user")):
                    #         breakpoint()
                    # print("e2")

                    hvr = [(hist_id, refined_history_prompt + hist_item) for hist_id, hist_item in refined_history_vec]
                    combined_history_vec = refined_history_vec
                    top_indices = retrieve_api_embedding((post_id, query), hvr, top_k=args.item_topk_collab, run_name=run_name, save=True, threshold=args.item_threshold,dataset=args.dataset)
                    # print("e3")

                    if "fine_grained" in prompt_type:
                        nid2items={}
                        tmp=[combined_history_vec[i] for i in top_indices]
                        for hist_id, hist_item in tmp:
                            cur_uid = hist_id.split("_")[0]
                            if cur_uid not in nid2items:
                                nid2items[cur_uid]=[]
                            nid2items[cur_uid].append((hist_id, hist_item))
                        cur_s=""
                        neighbor_uids = sorted(nid2items.keys())
                        for i, cur_uid in enumerate(neighbor_uids):
                            if not len(nid2items[cur_uid]): continue
                            cur_s+=f"===Neighbor {i+1} Profile===\n{ur.get_user_profile(cur_uid, user_data[cur_uid]['description'] if cur_uid in user_data else '')}\n"
                            cur_s+=f"===Neighbor {i+1} Data===\n"
                            for hist_id, hist_item in nid2items[cur_uid]:
                                cur_s+=f"{hist_item}\n"
                            # cur_s+="\n"
                        collab_history = cur_s
                    else: collab_history = "\n".join([combined_history_vec[i][1] for i in top_indices])
                    # print("e4")
                    # print(f"[collab_history]\n{collab_history}\n")

            # history_posts = [preprocess_tweet_link(item["text"]) for item in history_data[str(user_id)][:num_past]]
            # print("user_id", user_id)
            pred_fname, cur_prompt_prediction = make_llm_prediction(history, post_text, profile, prompt_type, use_refined_profile=args.use_refined_profile, augment=augment,choices=choices_text, collab_history=collab_history, args=args)

            """=================Get Eval=================="""
            cnt += 1
            if is_eval:
                res = load_file(pred_fname)
                # decoded_pred, sent_pred, intensity_pred = parse_output(res)
                sent_pred, explanation = parse_output(res)
                sent_pred = sent_pred.lower().strip().strip("\'").strip("\"")
                if sent_pred not in choices:
                    print("sent_pred", sent_pred)
                    print(f"error output sample {i}", res)
                sent_pred = choices[sent_pred] if sent_pred in choices else 0

                # sent_pred=choices[sent_pred.lower().strip()] if sent_pred.lower().strip() in choices else 0
                intensity_pred, decoded_pred = sent_pred, "a"

                # print(sent_pred, intensity_pred, decoded_pred)
                sent_preds.append(int(sent_pred))
                sent_labels.append(predicted)
                intensity_preds.append(int(intensity_pred))
                intensity_labels.append(abs(predicted_intensity - 3) if pure_intensity else predicted_intensity)
                decoded_preds.append(decoded_pred)
                decoded_labels_special.append([tgt_text])
                post_ids.append(post_id)

                # tmp_cur_prompt_prediction = load_file(args.prediction_prompt_dir + sample['tweet_id'] + ".txt")
                # num_tokens_prompt.append(len(gpt_tokenizer.encode(tmp_cur_prompt_prediction)))
                #
                # hist_content_prompt = tmp_cur_prompt_prediction[tmp_cur_prompt_prediction.find("[user historical"):]
                # # tmp_collab_history = collab_history if collab_history is not None else ""
                # num_tokens_hist.append(len(gpt_tokenizer.encode(hist_content_prompt)))

                if path_exists(args.prediction_prompt_dir + sample['tweet_id'] + ".txt"):
                    tmp_cur_prompt_prediction = load_file(args.prediction_prompt_dir + sample['tweet_id'] + ".txt")
                    num_tokens_prompt.append(len(gpt_tokenizer.encode(tmp_cur_prompt_prediction)))
                    hist_content_prompt = tmp_cur_prompt_prediction[tmp_cur_prompt_prediction.find("[user historical"):]
                    # tmp_collab_history = collab_history if collab_history is not None else ""
                    num_tokens_hist.append(len(gpt_tokenizer.encode(hist_content_prompt)))
                else:
                    tmp_cur_prompt_prediction=" "
                    num_tokens_prompt.append(10000)
                    num_tokens_hist.append(10000)

                # num_tokens_prompt.append(0)
                # num_tokens_hist.append(0)


        # exit()

        if is_eval:

            result_prediction = eval_tuple_response(sent_preds, sent_labels, intensity_preds, intensity_labels, decoded_preds, decoded_labels_special, num_tokens_prompt, num_tokens_hist, args.gen_eval ,post_ids)

            now = datetime.now()
            # current_time = now.strftime("%Y%m%d")
            current_time = now.strftime("%Y%m%d%H%M%S")

            # Example usage

            # outfn = f"{CNN_results_dir}chatgpt_case{case}_{prompt_type}_neighbor{args.neighbor_topk}_item{args.item_topk}_nh{args.n_hist}_ih{args.include_history}_en{args.include_empty_neighbors}_pint{pure_intensity}_{version}_{current_time}.json"
            mkdir(CNN_results_dir)

            output_base_name = f"chatgpt_case{case}_{prompt_type}_neighbor{args.neighbor_topk}_item{args.item_topk}_itemcb{args.item_topk_collab}_it{args.item_threshold}_nh{args.n_hist}_ih{args.include_hist}_en{args.include_empty_neighbors}_dp{args.do_process}_pint{pure_intensity}_{mdname}{version}"
            outfn = f"{CNN_results_dir}{output_base_name}_{current_time}.json"  # if args.case_study.strip() else ''
            # if args.case_study.strip() else ''
            print(cnt)
            print(outfn)
            print(" ".join([str(round(result_prediction[k], 2)) for k in eval_keys if k in result_prediction]))
            result_dict[args.case_study][prompt_type][version] = {keyname: round(result_prediction[keyname], 2) for keyname in eval_keys if keyname in result_prediction}  # result_prediction
            dump_file(result_prediction, outfn)

            file_path = f'{CNN_results_dir}result{args.dataset}.csv'  # Path to your CSV file
            df = load_or_create_dataframe(f'{CNN_results_dir}result{args.dataset}.csv', eval_keys)
            df = update_dataframe(df, file_path, current_time, case, prompt_type, args.neighbor_topk, args.neighbor_threshold, args.item_topk,args.item_topk_collab, args.item_threshold, 2500, args.n_hist, args.include_hist, args.include_empty_neighbors, args.do_process, pure_intensity,args.lc_model, args.nb_hist_only, "".join(args.modules), args.include_rf, version, {keyname: round(result_prediction[keyname], 2) for keyname in eval_keys if keyname in result_prediction})
            df.to_csv(file_path, index=False)

            # result_dict = result_dict.append({keyname: result_prediction[keyname] for keyname in eval_keys}, ignore_index=True)
