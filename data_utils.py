import json
import os 
import pandas as pd 
import numpy as np 


def convert_json_to_dataframe():

    filename = "data.json"

    with open(filename, "rb") as f:
        dataset = json.load(f)

    data = dataset["data"]

    df_all_data = pd.DataFrame()

    for list_idx, list_item in enumerate(data):
        title = list_item["title"]
   
        paragraphs = list_item["paragraphs"]
       
        for p_idx, paragraph in enumerate(paragraphs):
            paragraph_res = []
            
            context = paragraph["context"]

            for q_idx, qas in enumerate(paragraph["qas"]):
                question = qas["question"]
                is_impossible = qas["is_impossible"]
                answers = qas["answers"]
                question_id = qas["id"]
                
                # TODO: add plausible answers
                res = {"question": question, 
                    "list_idx": list_idx,
                    "paragraph_idx": p_idx,
                    "question_idx": q_idx,
                    "id": question_id, 
                    "is_impossible": is_impossible,
                    "answer_0": np.nan,
                    "answer_1": np.nan,
                    "answer_2": np.nan,
                    "answer_3": np.nan
                }
                
                if len(answers) > 0: 
                    for idx, answer in enumerate(answers): 
                        res[f"answer_{idx}"] = answer["text"]
                    
                paragraph_res.append(res)

            paragraph_df = pd.DataFrame(paragraph_res)
            paragraph_df["context"] = context
            paragraph_df["title"] = title
            
            df_all_data = pd.concat([df_all_data, paragraph_df], ignore_index=True)
    
    print("Saving dataframe of shape: ", df_all_data.shape)
    df_all_data.to_csv("dataset.csv", index=False)



def create_json_subset(df_sel):
    all_res = {"version": "v2.0", "data": []}

    all_res_list = []

    for list_idx in set(df_sel.list_idx):
        print(list_idx)

        df_idx = df_sel[df_sel.list_idx == list_idx]

        title = df_idx.title.iloc[0]

        print("title", title)

        paragraphs = []
        
        list_res = {"title": title, 
                    "paragraphs": paragraphs}
        
        paragraphs_idx = set(df_idx.paragraph_idx)
        
        print(paragraphs_idx)

        paragraphs_res = []

        for p_idx in paragraphs_idx:
            paragraph_df = df_idx[df_idx.paragraph_idx == p_idx]

            context = paragraph_df.context.iloc[0]

            paragraph_res = {"context": context, "qas": []}

            questions_df = paragraph_df[paragraph_df["id"].isin(qids)]

            if not questions_df.empty:
                question_idx = set(questions_df.question_idx)

                questions_list = []
            
                for q_idx in question_idx:
                    question_df = questions_df[questions_df.question_idx == q_idx]

                    question = question_df.question.iloc[0]
                    question_id = question_df["id"].iloc[0]
                    is_impossible = question_df.is_impossible.iloc[0]

                    answers = []

                    for i in range(0,3):
                        answer = question_df[f"answer_{i}"].iloc[0]

                        if answer == answer:
                        
                            answers.append({"text": answer})
                    
                    if is_impossible:
                        impossible_str = "true"
                    else:
                        impossible_str = "false"
                    
                    question_res = {"question": question,  
                                    "id": question_id,
                                    "answers": answers,
                                    "is_impossible": impossible_str
                                }
                    
                    questions_list.append(question_res)

                paragraph_res["qas"] = questions_list
                    
            
            paragraphs.append(paragraph_res)

        all_res_list.append(list_res)
            
    all_res["data"] = all_res_list

    with open("eval_results/data_updated_500.json", "w") as f:
        json.dump(all_res, f)


def merge_results():

    # TODO: remove redundant code, replace with for loop 
    filename = "f1_thresh_by_qid.json"
    with open(filename, "rb") as f:
        f1_scores = json.load(f)
        
    df_scores = pd.DataFrame.from_dict(f1_scores, orient="index", columns=["f1_score"])
    df_scores.index.name = "id"
    df_scores.reset_index(inplace=True, drop=False)
    
    filename = "exact_thresh_by_qid.json"
    with open(filename, "rb") as f:
        exact_scores = json.load(f)
        
    df_scores_exact = pd.DataFrame.from_dict(exact_scores, orient="index", columns=["exact_score"])
    df_scores_exact.index.name = "id"
    df_scores_exact.reset_index(inplace=True, drop=False)
    

    filename = "pred.json"
    with open(filename, "rb") as f:
        pred = json.load(f)
        
    df_pred = pd.DataFrame.from_dict(pred, orient="index", columns=["pred"])
    df_pred.index.name = "id"
    df_pred.reset_index(inplace=True, drop=False)
    
    df = pd.read_csv("df_questions_all.csv")
    df = df.merge(df_scores, how="left", on="id")
    df = df.merge(df_scores_exact, how="left", on="id")
    df = df.merge(df_pred, how="left", on="id")

    return df 


def collect_all_results():

    all_res = []

    files = os.listdir("eval_results")
    files = [f for f in files if f.startswith("eval_pred_500") and f.endswith("json")]

    for f in files:
        filepath = os.path.join(os.getcwd(), "eval_results", f) 
       
        with open(filepath, "rb") as file_to_load:
            res = json.load(file_to_load)
        
        res["experiment"] = f

        df_res = pd.DataFrame.from_dict({k: [v] for k, v in res.items()}, orient="columns")
      
        all_res.append(df_res)
    
    df_all = pd.concat(all_res, ignore_index=True)

    return df_all