import os 
import sys 
import json 
import pandas as pd 


dir_path = sys.argv[1]
output_dir = sys.argv[2]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def get_source_id(name):
    return "_".join(name.split("_")[:2])

if __name__ == "__main__":

    files = [x for x in os.listdir(dir_path) if x.endswith("json")]
    df = pd.DataFrame(files, columns=["name"])

    df["id"] = df["name"].apply(get_source_id)

    for group_id, summary_ids in df.groupby("id"):
        print(group_id)
        first_summary, second_summary = summary_ids.iloc[0]["name"], summary_ids.iloc[1]["name"]
        with open(os.path.join(dir_path, first_summary), 'r') as f:
            s1 = json.load(f)
        with open(os.path.join(dir_path, second_summary), 'r') as f:
            s2 = json.load(f)
        data = {
            "sourceId": group_id,
            "source": s1["source"],
            "summaries": [
                {
                    "tokens": s1["summary"],
                    "spans": s1["spans"],
                    "qas": s1["qas"],
                    "label": s1["label"],
                    "summaryId": first_summary.split(".")[0].split("_")[-1],
                    "predicates": [i for i, x in enumerate(s1["spans"]) if x["predicate"] == True]
                },
                {
                    "tokens": s2["summary"],
                    "spans": s2["spans"],
                    "qas": s2["qas"],
                    "label": s2["label"],
                    "summaryId": second_summary.split(".")[0].split("_")[-1],
                    "predicates": [i for i, x in enumerate(s2["spans"]) if x["predicate"] == True]
                }
            ],
            "datasource": "mediasum",
            "dataset": "tofueval",
            
        }
        with open(os.path.join(output_dir, f"{group_id}.json"), "w") as f:
            json.dump(data, f, indent=4)
        
                