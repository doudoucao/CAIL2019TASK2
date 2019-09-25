import json
import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer
from BertForMultiLabelClassification import BertForMultiLabelClassification
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from data_process import *
from tqdm import tqdm
import numpy as np
from run_multilabelclassifier import MultiLabelTextProcessor, convert_examples_to_features

input_path_labor = "/input/labor/input.json"
input_path_divorce = "/input/divorce/input.json"
# input_path_divorce = "./data/data_small/divorce/data_small_selected.json"
input_path_loan = "/input/loan/input.json"
output_path_labor = "/output/labor/output.json"
output_path_divorce = "/output/divorce/output.json"
# output_path_divorce = 'output/output.json'
output_path_loan = "/output/loan/output.json"
tag_path_labor = 'data/data_small/labor/tags.txt'
tag_path_divorce = 'data/data_small/divorce/tags.txt'
tag_path_loan = 'data/data_small/loan/tags.txt'


def predict(args, model, tokenizer, data_dir, test_filename, input_path, output_path, tagname_dic):
    ## set GPU
    if args["local_rank"] == -1 or args["no_cuda"]:
        device = torch.device("cuda" if torch.cuda.is_available() and not args["no_cuda"] else "cpu")
        n_gpu = torch.cuda.device_count()
    # n_gpu = 1
    else:
        torch.cuda.set_device(args['local_rank'])
        device = torch.device("cuda", args['local_rank'])
        n_gpu = 1

    predict_processor = MultiLabelTextProcessor(data_dir)
    test_examples = predict_processor.get_test_examples(data_dir, test_filename, size=-1)

    labels = predict_processor.get_labels()

    # input_data = [{'id': input_example.guid, 'comment_text': input_example.text_a} for input_example in test_examples]

    test_features = convert_examples_to_features(test_examples, labels, args['max_seq_length'], tokenizer)

    print("***** Running prediction *****")
    print("  Num examples = %d"% len(test_examples))
    print("  Batch size = %d"% args['eval_batch_size'])

    all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)

    test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

    # run prediction for full data
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args['eval_batch_size'])

    all_logits = None
    model.to(device)
    model.eval()
    nb_eval_steps, nb_eval_examples = 0, 0
    for step, batch in enumerate(tqdm(test_dataloader,desc="Prediction Iteration" )):
        input_ids, input_mask, segment_ids = batch
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)
            logits = logits.sigmoid()

        if all_logits is None:
            all_logits = logits.detach().cpu().numpy()
        else:
            all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    inf = open(input_path, "r", encoding='utf-8')
    ouf = open(output_path, "w", encoding='utf-8')

    i = 0
    for line in inf:
        pre_doc = json.loads(line)
        new_pre_doc = []
        for sent in pre_doc:
            sent['labels'] = []  # 将该空列表替换成你的模型预测的要素列表结果
            labels = [j for j, x in enumerate(all_logits[i]) if x > 0.5]
            # print(labels)
            for m in labels:
                sent['labels'].append(tagname_dic[m])
                # print(tagname_dic[m])
            new_pre_doc.append(sent)
            i += 1
        json.dump(new_pre_doc, ouf, ensure_ascii=False)
        ouf.write('\n')

    inf.close()
    ouf.close()


def main():
    args = {
        "train_size": -1,
        "val_size": -1,
        "data_dir": './data/data_small/loan',
        "task_name": "multilabel",
        "no_cuda": False,
        "bert_model": '/media/iiip/A343-9833/CZX/bert-base-chinese',
        "output_dir": 'bert_loan',
        "max_seq_length": 160,
        "do_train": False,
        "do_eval": True,
        "do_lower_case": True,
        "train_batch_size": 32,
        "eval_batch_size": 32,
        "learning_rate": 5e-5,
        "num_train_epochs": 4.0,
        "warmup_proportion": 0.1,
        "local_rank": -1,
        "seed": 42,
        "gradient_accumulation_steps": 1,
        "optimize_on_cpu": False,
        "fp16": False,
        "loss_scale": 128
    }

    # ######  divorce predict #####
    divorce_tag_dic, divorce_tagname_dic = init(tag_path_divorce)
    num_divorce_labels = len(divorce_tag_dic)
    read_trainData(input_path_divorce, tag_path_divorce, './data/data_small/divorce/divorce_test.txt')
    tokenizer = BertTokenizer.from_pretrained('./bert_divorce', do_lower_case=args['do_lower_case'])
    model = BertForMultiLabelClassification.from_pretrained('./bert_divorce', num_labels=num_divorce_labels)
    predict(args, model, tokenizer, data_dir='./data/data_small/divorce', test_filename='divorce_test.txt',
            input_path=input_path_divorce, output_path=output_path_divorce, tagname_dic=divorce_tagname_dic)

    # ##### labor predict ###
    labor_tag_dic, labor_tagname_dic = init(tag_path_labor)
    num_labor_labels = len(labor_tag_dic)

    read_trainData(input_path_labor, tag_path_labor, './data/data_small/labor/labor_test.txt')
    tokenizer = BertTokenizer.from_pretrained('./bert_labor', do_lower_case=args['do_lower_case'])
    model = BertForMultiLabelClassification.from_pretrained('./bert_labor', num_labels=num_labor_labels)
    predict(args, model, tokenizer, data_dir='./data/data_small/labor', test_filename='labor_test.txt',
            input_path=input_path_labor, output_path=output_path_labor, tagname_dic=labor_tagname_dic)

    # ##### loan predict ###
    loan_tag_dic, loan_tagname_dic = init(tag_path_loan)
    num_loan_labels = len(loan_tag_dic)

    read_trainData(input_path_loan, tag_path_loan, './data/data_small/loan/loan_test.txt')
    tokenizer = BertTokenizer.from_pretrained('./bert_loan', do_lower_case=args['do_lower_case'])
    model = BertForMultiLabelClassification.from_pretrained('./bert_loan', num_labels=num_loan_labels)
    predict(args, model, tokenizer, data_dir='./data/data_small/loan', test_filename='loan_test.txt',
            input_path=input_path_loan, output_path=output_path_loan, tagname_dic=loan_tagname_dic)


if __name__ == '__main__':
    main()