from torch.utils.data import DataLoader
import torch
import torch.utils.data
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, mean_squared_error, mean_absolute_error, \
    precision_score, recall_score, cohen_kappa_score
import numpy as np
# from data import collate_wrapper
# from pprint import pprint as pp
# from sklearn.metrics import accuracy_score
# from train_utils import seed_worker

from datasets import load_dataset, load_metric
# import numpy as npeval_final.py
# from transformers import DataCollatorForSeq2Seq, DataCollatorWithPadding
# from datasets import Dataset, load_dataset
# from torch.utils.data.dataloader import DataLoader
# from glob import glob
# from data import pad_to_batch# , CustomBatch, collate_wrapper
from pprint import pprint as pp
from tqdm import tqdm
from eval_final import Evaluate
from rouge import Rouge
# import wandb
from IPython import embed
from utils.utils import *
from utils.data_utils import *
from scipy.stats import wasserstein_distance



from scipy.stats import pearsonr, spearmanr


def get_prf(targets, preds, average="micro", verbose=False):
    precision = precision_score(targets, preds, average=average)
    recall = recall_score(targets, preds, average=average)
    f1 = f1_score(targets, preds, average=average)
    if verbose: print(f"{average}: precision {precision} recall {recall} f1 {f1}")
    return precision, recall, f1


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def convert_to_dict(ls):
    # ls = [self[i] for i in range(len(self))]
    keys = list(ls[0].keys())
    ress = {}
    for key in keys:
        ress[key] = [i[key] for i in ls]

    # print("\nres.keys()", res.keys())
    return ress
#
# def gen_metric_compute(preds, labels):
#     result = metric.compute(predictions=decoded_preds, references=decoded_labels)
#     result = {"bleu": result["score"]}
#
#     result2 = metric2.compute(predictions=decoded_preds, references=[sent[0] for sent in decoded_labels], lang="en")
#     result["bertscore"] = result2['f1'] / len(result2['f1'])


# class GenerationEvaluator:
#     def __init__(self):
#         self.sacrebleu= load_metric("sacrebleu")
#         self.bertscore= load_metric("bertscore")
#         self.general=Evaluate()
#
#     def compute(self, preds, references, dev=True):
#
#         ## BLEU's
#
#         preds_alt = {i: item for i, item in enumerate(preds)}  # dict style
#         references_alt = [item[0] for item in references]  # no streaming style
#         final_scores = self.general.evaluate(live=True, cand=preds_alt, ref=references)
#         # ## Rouge
#         # rouge = Rouge()
#         # # final_scores.update(rouge.get_scores(preds, references_alt, avg=True))
#         # print(rouge.get_scores(preds, references_alt, avg=True))
#         ## SacreBleu
#         result = self.sacrebleu.compute(predictions=preds, references=references)
#         final_scores["sacrebleu"] = result['score']
#
#         result = self.bertscore.compute(predictions=preds, references=[sent[0] for sent in references], lang="en")
#         final_scores["bertscore"] = sum(result2['f1']) / len(result['f1'])
#         # final_scores['epoch']=0
#         return final_scores

        # if get_scores:
        #     return final_scores

def get_mrr(preds, labels, verbose=False):
    total_score = 0
    score_cnt = 0
    for i, (pl, tl) in enumerate(zip(preds, labels)):
        assert len(tl), breakpoint()
        if tl:
            score_cnt += 1
            if tl[0] in pl:
                total_score += 1 / (pl.index(tl[0]) + 1)

    return round(total_score / score_cnt * 100, 4)

def get_scores_multilabel_clf(preds, labels, verbose=False):
    # print("get_scores_multilabel_clf")
    # print("logits, labels", logits, labels)
    # preds = (torch.sigmoid(get_tensor_float(logits)) > 0.5).int().tolist()

    score_dict = {}
    mi_precision, mi_recall, mi_f1 = get_prf(labels, preds, average="micro", verbose=False)
    ma_precision, ma_recall, ma_f1 = get_prf(labels, preds, average="macro", verbose=False)
    #get pearson and spearman

    pearson = pearsonr(preds, labels)[0]
    # print(list(zip(preds, labels)))
    spearman = spearmanr(preds, labels)[0]
    kappa = cohen_kappa_score(preds, labels, weights="quadratic")
    if np.isnan(pearson):
        pearson = 0
    if np.isnan(spearman):
        spearman = 0
    if np.isnan(kappa):
        kappa = 0
    score_dict.update({
        "mif1": mi_f1,
        "maf1": ma_f1,
        "accuracy": accuracy_score(labels, preds),
        "miprecision": mi_precision,
        "mirecall": mi_recall,
        "maprecision": ma_precision,
        "marecall": ma_recall,
        "pearson": pearson,
        "spearman": spearman,
        "kappa": kappa,
    })
    for key in score_dict:
        score_dict[key] = round(score_dict[key]*100, 4)
    return score_dict





def get_scores_binary_clf(preds, labels, num_decimals=4):
    # preds = np.argmax(logits, axis=-1)

    score_dict = {}
    try:
        precision, recall, f1 = get_prf(labels, preds, average="binary", verbose=False)
    except Exception as e:
        print(e)
        embed()
    score_dict.update({
        "f1": round(f1, num_decimals),
        "accuracy":  round(accuracy_score(labels, preds), num_decimals),
        "precision":  round(precision, num_decimals),
        "recall":  round(recall, num_decimals),
    })
    return score_dict



def get_prf(targets, preds, average="micro", verbose=False):
    precision = precision_score(targets, preds, average=average, zero_division=0)
    recall = recall_score(targets, preds, average=average, zero_division=0)
    f1 = f1_score(targets, preds, average=average, zero_division=0)
    if verbose: print(f"{average}: precision {precision} recall {recall} f1 {f1}")
    return precision, recall, f1


def generate_steps(args, model, data, tokenizer, data_collator=None, no_scoring=False):
    pass


def eval_tuple_response(sent_preds, sent_labels, intensity_preds, intensity_labels, decoded_preds,decoded_labels_special,num_tokens_prompt,num_tokens_hist, gen_eval=1, post_ids=None):

    # metric2: Any = load_metric("bertscore")
    # ppl_metric: Any = load_metric("meteor")

    result={}
    clf_score_dict = get_scores_multilabel_clf(sent_preds, sent_labels)
    for key in ["pearson", "spearman", "kappa"]:
        clf_score_dict.pop(key, None)
    # clf_score_dict=modify_dict_keys(clf_score_dict, "intensity_")
    result.update(clf_score_dict)

    clf_score_dict = get_scores_multilabel_clf(intensity_preds, intensity_labels)
    clf_score_dict = {key: value for key, value in clf_score_dict.items() if key in ["pearson", "spearman", "kappa"]}
    # clf_score_dict=modify_dict_keys(clf_score_dict, "intensity_")
    result.update(clf_score_dict)

    # mse error
    result["mse"] = round(mean_squared_error(intensity_labels, intensity_preds), 3)
    result["mae"] = round(mean_absolute_error(intensity_labels, intensity_preds), 3)

    #1-Wasserstein distance
    print("intensity_labels",intensity_labels)
    print("len post_ids",len(set(post_ids)))
    print("set intensity_labels",set(intensity_labels))
    # embed()
    # group intensity_labels by post_id into a dict
    post_id_to_intensity_labels = {post_id: [] for post_id in post_ids}
    post_id_to_intensity_preds = {post_id: [] for post_id in post_ids}
    for post_id, intensity_label, intensity_pred in zip(post_ids, intensity_labels, intensity_preds):
        if intensity_label in [99,8]:
            continue
        post_id_to_intensity_labels[post_id].append(intensity_label)
        post_id_to_intensity_preds[post_id].append(intensity_pred)
    tmpp=sum([1-(wasserstein_distance(post_id_to_intensity_labels[key], post_id_to_intensity_preds[key])/6) for key in post_id_to_intensity_labels])/len(post_id_to_intensity_labels)
    result["wasserstein"] = round(tmpp* 100, 4)

    if gen_eval:
        # Bleu
        metric: Any = load_metric("sacrebleu")
        result_tmp = metric.compute(predictions=decoded_preds, references=decoded_labels_special)
        result_tmp = {"bleu": result_tmp["score"]}
        result.update(result_tmp)

        eval_f: Any = Evaluate()
        decoded_preds_alt = {i: item for i, item in enumerate(decoded_preds)}  # dict style
        result.update(eval_f.evaluate(live=True, cand=decoded_preds_alt, ref=decoded_labels_special))

        # Rouge
        # decoded_preds_alt = {i: item for i, item in enumerate(decoded_preds)}  # dict style
        # result.update(self.eval_f.evaluate(live=True, cand=decoded_preds_alt, ref=decoded_labels))
        decoded_labels = [sent[0] for sent in decoded_labels_special]
        rouge_metric: Any = load_metric("rouge")
        rouge = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
        rouge = {key: round(value.mid.fmeasure * 100, 1) for key, value in rouge.items()}
        result.update(rouge)

        # Meteor
        meteor_metric: Any = load_metric("meteor")
        meteor = meteor_metric.compute(predictions=decoded_preds, references=decoded_labels)
        meteor["meteor"] *= 100
        result.update(meteor)

    same_vec=(np.array(sent_preds)==np.array(sent_labels))
    avg_prompt_tokens=np.mean(num_tokens_prompt)
    avg_hist_tokens=np.mean(num_tokens_hist)

    result.update({
        "pi_prompt": result["mif1"]/avg_prompt_tokens,
        "pi_hist": result["mif1"]/avg_hist_tokens,
        "avg_prompt_tokens": avg_prompt_tokens,
        "avg_hist_tokens": avg_hist_tokens,
        "pi_prompt2": np.mean(same_vec*np.array(num_tokens_prompt)),
        "pi_hist2": np.mean(same_vec*np.array(num_tokens_hist))
    })

    return result
