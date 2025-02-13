
from nltk.corpus import stopwords
import spacy
# from gensim.parsing.preprocessing import remove_stopwords, STOPWORDS

all_month_oneie_dir="data/sent_track/allmonth_oneie/"

mftc_dir="data/MFTC/"
download_dir="D:/Downloads/"
mftc_data_path="data/MFTC/MFTC_V4_text.json"
mftc_sandy_dir=f"{mftc_dir}Sandy/"
mftc_all_dir=f"{mftc_dir}All/"
mftc_allfreval_dir=f"{mftc_dir}All_fr_eval/"
mftc_All_fr_eval_dir=f"{mftc_dir}All_fr_eval/"
mftc_fr_path=f"{mftc_allfreval_dir}sampled_twitter_preview.json"
fr_raw_data_path=f"{mftc_allfreval_dir}sampled_twitter_preview.json"

user_mvp_dir="data/user_mvp/"
user_mvp_dir_all="data/user_mvp/full/"

sent_track_dir="data/sent_track/"

NS_tmp_dir="..\\data\WikiHow\\tmp\\"
NS_tmp_dir_filtered="..\\data\WikiHow\\tmp\\filtered\\"
NS_cgen_dir="data/wikihow/full_cgen/"
NS_ss9_dir="data/wikihow/subset9/"
NS_ss9_grounded_dir=f"{NS_ss9_dir}grounded/"
NS_ss9_train=f"{NS_ss9_dir}data_train.json"

data_dir="data/"
# CNN_dir=f"{data_dir}CNN/debug/"
CNN_dir=f"{data_dir}CNN/"
# CNN_dir=f"{data_dir}OpinionQA/human_resp/"
# CNN_dir=f"{data_dir}OpinionQA/human_resp/American_Trends_Panel_W34/"
# CNN_dir=f"{data_dir}OpinionQA/human_resp/American_Trends_Panel_W41/"
# CNN_dir=f"{data_dir}OpinionQA/human_resp/American_Trends_Panel_W82/"

CNN_reply_dir=f"{CNN_dir}reply/"
CNN_user_info_dir=f"{CNN_dir}reply/"
CNN_train_filename=f"{CNN_dir}response_data_balanced_train.json"
CNN_dev_filename=f"{CNN_dir}response_data_balanced_dev.json"
CNN_test_filename=f"{CNN_dir}response_data_balanced_test_anno.json"
CNN_train_anno_filename=f"{CNN_dir}response_data_balanced_train_anno.json"
CNN_dev_anno_filename=f"{CNN_dir}response_data_balanced_dev_anno.json"
CNN_test_anno_filename=f"{CNN_dir}response_data_balanced_test_anno.json"
CNN_post_filename=f"{CNN_dir}post_data.json"
CNN_users_info_filename=f"{CNN_dir}users_info_dict_a.json"
CNN_users_history_filename=f"{CNN_dir}users_history_dict_a.json"
CNN_users_info_filename=f"{CNN_dir}users_info_dict.json"
CNN_users_history_filename=f"{CNN_dir}users_history_dict.json"
# CNN_users_info_filename=f"{CNN_dir}users_info_dict.json"
# CNN_users_history_filename=f"{CNN_dir}users_history_dict.json"
CNN_users_followings_dict_filename=f"{CNN_dir}users_followings_dict.json"
CNN_users_embed_dir=f"{CNN_dir}user_embeddings/"
CNN_id_to_index=f"{CNN_dir}id_to_index.json"
CNN_users_info_r_filename=f"{CNN_dir}users_info_dict_with_chatgptanno_r.json"
CNN_users_info_rf_filename=f"{CNN_dir}users_info_dict_with_chatgptanno_rf.json"
CNN_users_info_sc_filename=f"{CNN_dir}users_info_dict_with_chatgptanno_sc.json"
CNN_users_info_l_filename=f"{CNN_dir}users_info_dict_with_chatgptanno.json"
CNN_train_lurker_filename=f"{CNN_dir}train_anno_case_lurker.json"
CNN_dev_anno_lurker_filename=f"{CNN_dir}dev_anno_case_lurker.json"
CNN_test_anno_lurker_filename=f"{CNN_dir}test_anno_case_lurker.json"
CNN_api_embeddings_dir=f"{CNN_dir}api_embeddings/"
CNN_r_anno_json_dir=f"{CNN_dir}chatgptanno_r_json/"
CNN_rf_anno_json_dir=f"{CNN_dir}chatgptanno_rf_json/"
CNN_prediction_dir=f"{CNN_dir}predictions/"
CNN_prediction_prompt_dir=f"{CNN_dir}prediction_prompts/"
CNN_results_dir=f"{CNN_dir}results/"
CNN_non_empty_hist_uids_filename=f"{CNN_dir}non_empty_hist_uids.json"
CNN_non_empty_hist_uids_processed_filename=f"{CNN_dir}non_empty_hist_uids_processed.json"
# CNN_neighbor_ids_dir=f"{CNN_api_embeddings_dir}user_meta_key_retrieval_uids/"

# users_history_filename=f"users_history_dict.json"
# train_anno_filename=f"response_data_balanced_train_anno.json"
# dev_anno_filename=f"response_data_balanced_dev_anno.json"
# test_anno_filename=f"response_data_balanced_test_anno.json"
# users_info_r_filename=f"users_info_dict_with_chatgptanno_r.json"
# users_info_rf_filename=f"users_info_dict_with_chatgptanno_rf.json"
# api_embeddings_dir=f"api_embeddings/"
# r_anno_json_dir=f"chatgptanno_r_json/"
# rf_anno_json_dir=f"chatgptanno_rf_json/"
# prediction_dir=f"predictions/"
# results_dir=f"results/"


user_meta_keys=["value", "ideolog", "topic","profession","possession","role"]
desc_keys = ["human value", "moral value", "ideolog", "topic", "role", "pattern", "analysis", "note", "possess", "profess"]
list_keys = ["interested issue", "interested entit"]

metric_keys=sorted({"mif1": 1, "maf1": 1, "accuracy": 1, "miprecision": 1, "mirecall": 2, "maprecision": 45, "marecall": 1, "pearson": 1,
             "spearman": 1, "kappa": 1, "bleu": 1, "Bleu_1": 0, "Bleu_2": 3, "Bleu_3":1,
             "Bleu_4": 1, "rouge1": 11.9, "rouge2":0, "rougeL": 9.9, "rougeLsum": 9.9, "meteor": 7.}.keys())

history_prompt = "Encode the user's historical tweet for retrieval. Even if not directly related to a news headline, this tweet may still hint at the user's opinions and personality. Consider the tweet carefully. Tweet: "
refined_history_prompt = "Encode the analysis on the user for retrieval. Even if not directly related to a news headline, this analysis on the user may still hint at the user's opinions and personality. Consider the analysis carefully. Analysis: "


"""==========================================Metadata=============================================="""


HUMAN_VALUES=["conformity", "tradition", "security", "power", "achievement", "hedonism", "stimulation", "self-direction", "universalism", "benevolence"]
MORAL_VALUES=["authority", "betrayal", "care", "cheating", "degradation", "fairness", "harm", "loyalty", "purity", "subversion"]

analysis_dir=f"../INCAS/analysis/"

BBC_dir="data/BBC/"
BBC_reply_dir="data/BBC/reply/"

# ALL_STOPWORDS = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat']).Defaults.stop_words
# ALL_STOPWORDS.update(set(stopwords.words('english')))
# ALL_STOPWORDS.update(set(STOPWORDS))

abbr2category = {"food": "Food and Entertaining",
                 "rel": "Relationships", "home": "Home and Garden",
                 "cv": "Cars & Other Vehicles", "ec": "Education and Communications",
                 "pa": "Pets and Animals", "ae": "Arts and Entertainment", "ps": "Personal Care and Style", "he": "Health", "fb": "Finance and Business",
                 "hc": "Hobbies and Crafts"}
