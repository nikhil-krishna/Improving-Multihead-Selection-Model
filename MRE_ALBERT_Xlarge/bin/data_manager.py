import os
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../bert")))
import tokenization

print("if not have raw data, please dowload data from http://lic2019.ccf.org.cn/kg !")

def unzip_and_move_files():
    "Unzip the original file, store it in folder /raw_data"
    os.system("unzip dev_data.json.zip")
    os.system("mv dev_data.json raw_data/dev_data.json")
    os.system("unzip train_data.json.zip")
    os.system("mv train_data.json raw_data/train_data.json")


class Model_data_preparation(object):

    def __init__(self, RAW_DATA_INPUT_DIR="raw_data", DATA_OUTPUT_DIR="classfication_data",
                 vocab_file_path="vocab.txt", do_lower_case=True, Competition_Mode=False, Valid_Model=False):
        # BERT: contains WordPiece tool. Slice Chinese into single characters. 
        self.bert_tokenizer = tokenization.FullTokenizer(vocab_file=self.get_vocab_file_path(vocab_file_path),
                                                         do_lower_case=do_lower_case)  # initialized bert_token
        self.DATA_INPUT_DIR = self.get_data_input_dir(RAW_DATA_INPUT_DIR)
        self.DATA_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), DATA_OUTPUT_DIR)
        self.Competition_Mode = Competition_Mode
        self.Valid_Model= Valid_Model
        print("Data input dir：", self.DATA_INPUT_DIR)
        print("Data output dir：", self.DATA_OUTPUT_DIR)
        print("output valid data：", self.Competition_Mode)
        print("Output test data：", self.Valid_Model)

    # Get input path
    def get_data_input_dir(self, RAW_DATA_INPUT_DIR):
        DATA_INPUT_DIR = os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../")), RAW_DATA_INPUT_DIR)
        return DATA_INPUT_DIR

    # Get voc path
    def get_vocab_file_path(self, vocab_file_path):
        vocab_file_path = os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../pretrained_model/chinese_L-12_H-768_A-12")), vocab_file_path)
        return vocab_file_path

    def subject_object_labeling(self, spo_list, text_tokened, bert_tokener_error_log_f):
        def _index_q_list_in_k_list(q_list, k_list):
            """Known q_list in k_list, find index(first time) of q_list in k_list"""
            q_list_length = len(q_list)
            k_list_length = len(k_list)
            for idx in range(k_list_length - q_list_length + 1):
                t = [q == k for q, k in zip(q_list, k_list[idx: idx + q_list_length])]
                # print(idx, t)
                if all(t):
                    # print(idx)
                    idx_start = idx
                    return idx_start

        def _labeling_type(subject_object, so_type):
            tokener_error_flag = False
            so_tokened = self.bert_tokenizer.tokenize(subject_object)
            so_tokened_length = len(so_tokened)
            idx_start = _index_q_list_in_k_list(q_list=so_tokened, k_list=text_tokened)
            if idx_start is None:
                tokener_error_flag = True
                '''
                实体: "1981年"  原句: "●1981年2月27日，中国人口学会成立"
                so_tokened ['1981', '年']  text_tokened ['●', '##19', '##81', '年', '2', '月', '27', '日', '，', '中', '国', '人', '口', '学', '会', '成', '立']
                so_tokened 无法在 text_tokened 找到！原因是bert_tokenizer.tokenize 分词增添 “##” 所致！
                '''
                bert_tokener_error_log_f.write(str(so_tokened) + " @@ " + str(text_tokened) + "\n")
            else:
                labeling_list[idx_start] = "B-" + so_type
                if so_tokened_length == 2:
                    labeling_list[idx_start + 1] = "I-" + so_type
                elif so_tokened_length >= 3:
                    labeling_list[idx_start + 1: idx_start + so_tokened_length] = ["I-" + so_type] * (so_tokened_length - 1)
            return idx_start, tokener_error_flag

        labeling_list = ["O" for _ in range(len(text_tokened))]
        predicate_value_list = [[] for _ in range(len(text_tokened))]
        predicate_location_list = [[] for _ in range(len(text_tokened))]

        tokener_error_flag = False
        for spo_item in spo_list:
            subject = spo_item["subject"]
            subject_type = spo_item["subject_type"]
            object = spo_item["object"]
            object_type = spo_item["object_type"]
            predicate_value = spo_item["predicate"]
            subject_idx_start, flag_A = _labeling_type(subject, subject_type)
            object_idx_start, flag_B = _labeling_type(object, object_type)
            if flag_A or flag_B:
                tokener_error_flag = True
                return labeling_list,predicate_value_list, predicate_location_list, tokener_error_flag
            predicate_value_list[subject_idx_start].append(predicate_value)
            predicate_location_list[subject_idx_start].append(object_idx_start)

        
        for idx in range(len(text_tokened)):
            if len(predicate_value_list[idx]) == 0:
                predicate_value_list[idx].append("N") 
            if len(predicate_location_list[idx]) == 0:
                predicate_location_list[idx].append(idx)

        for idx, token in enumerate(text_tokened):
            """标注被 bert_tokenizer.tokenize 拆分的词语"""
            if token.startswith("##"):
                labeling_list[idx] = "[##WordPiece]"

        return labeling_list,predicate_value_list, predicate_location_list, tokener_error_flag

    def separate_raw_data_and_token_labeling(self):
        if not os.path.exists(self.DATA_OUTPUT_DIR):
            os.makedirs(os.path.join(self.DATA_OUTPUT_DIR, "train"))
            os.makedirs(os.path.join(self.DATA_OUTPUT_DIR, "valid"))
            os.makedirs(os.path.join(self.DATA_OUTPUT_DIR, "test"))

        file_set_type_list = ["train", "valid", "test"]
        if self.Valid_Model:
            file_set_type_list = ["test"]
        for file_set_type in file_set_type_list:
            print("produce data will store in: ", os.path.join(os.path.join(self.DATA_OUTPUT_DIR, file_set_type)))
            if file_set_type in ["train", "valid"] or not self.Competition_Mode:
                labeling_out_f = open(
                    os.path.join(os.path.join(self.DATA_OUTPUT_DIR, file_set_type), "labeling_out.txt"), "w",
                    encoding='utf-8')
                predicate_value_out_f = open(
                    os.path.join(os.path.join(self.DATA_OUTPUT_DIR, file_set_type), "predicate_value_out.txt"), "w",
                    encoding='utf-8')
                predicate_location_out_f = open(
                    os.path.join(os.path.join(self.DATA_OUTPUT_DIR, file_set_type), "predicate_location_out.txt"), "w",
                    encoding='utf-8')
                bert_tokener_error_log_f = open(
                    os.path.join(os.path.join(self.DATA_OUTPUT_DIR, file_set_type), "bert_tokener_error_log.txt"), "w",
                    encoding='utf-8')
            text_f = open(os.path.join(os.path.join(self.DATA_OUTPUT_DIR, file_set_type), "text.txt"), "w",
                          encoding='utf-8')
            token_in_f = open(os.path.join(os.path.join(self.DATA_OUTPUT_DIR, file_set_type), "token_in.txt"), "w",
                              encoding='utf-8')
            token_in_not_UNK_f = open(
                os.path.join(os.path.join(self.DATA_OUTPUT_DIR, file_set_type), "token_in_not_UNK.txt"), "w",
                encoding='utf-8')

            if file_set_type == "train":
                path_to_raw_data_file = "train_data.json"
            elif file_set_type == "valid":
                path_to_raw_data_file = "dev_data.json"
            else:
                if self.Competition_Mode == True:
                    path_to_raw_data_file = "test1_data_postag.json"
                else:
                    path_to_raw_data_file = "dev_data.json"

            with open(os.path.join(self.DATA_INPUT_DIR, path_to_raw_data_file), 'r', encoding='utf-8') as f:
                count_numbers = 0
                while True:
                    line = f.readline()
                    if line:
                        count_numbers += 1
                        a_row_raw_data = json.loads(line)
                        if (not self.Competition_Mode) or file_set_type in ["train", "valid"]:
                            spo_list = a_row_raw_data["spo_list"]
                        else:
                            spo_list = []
                        text = a_row_raw_data["text"]
                        text_tokened = self.bert_tokenizer.tokenize(text)
                        text_tokened_not_UNK = self.bert_tokenizer.tokenize_not_UNK(text)
                        if file_set_type in ["train", "valid"] or (not self.Competition_Mode):
                            labeling_list, predicate_value_list, predicate_location_list, tokener_error_flag = \
                                self.subject_object_labeling(spo_list=spo_list, text_tokened=text_tokened,
                                                             bert_tokener_error_log_f=bert_tokener_error_log_f)
                            if tokener_error_flag == False:
                                labeling_out_f.write(" ".join(labeling_list) + "\n")
                                predicate_value_out_f.write(str(predicate_value_list) + "\n")
                                predicate_location_out_f.write(str(predicate_location_list) + "\n")
                                text_f.write(text + "\n")
                                token_in_f.write(" ".join(text_tokened) + "\n")
                                token_in_not_UNK_f.write(" ".join(text_tokened_not_UNK) + "\n")
                        else:
                            text_f.write(text + "\n")
                            token_in_f.write(" ".join(text_tokened) + "\n")
                            token_in_not_UNK_f.write(" ".join(text_tokened_not_UNK) + "\n")
                    else:
                        break
            print("all numbers", count_numbers)
            print("\n")

if __name__ == "__main__":
    RAW_DATA_DIR = "raw_data"
    DATA_OUTPUT_DIR = "standard_format_data"
    Competition_Mode = True
    Valid_Mode = False
    model_data = Model_data_preparation(
        RAW_DATA_INPUT_DIR=RAW_DATA_DIR, DATA_OUTPUT_DIR=DATA_OUTPUT_DIR, Competition_Mode=Competition_Mode, Valid_Model=Valid_Mode)
    model_data.separate_raw_data_and_token_labeling()
