import os 
import pandas as pd 
from tqdm import tqdm 
import spacy 
from itertools import chain
import collections 
import json 
from spacy.tokens import Doc
import jsonlines
from qasem_parser import QasemParser, QasemFrame, QasemArgument
import argparse 
from loc_unfaith import LocUnfaith

tqdm.pandas()

def get_spacy_summary(summary):
    return Doc.from_docs(list(nlp.pipe(summary)))


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--data_path", 
    type=str, 
    help='path to jsonl file where each row includes at least'
      'the fields `summary`, `article`, `datasource` and `label` (or `labels` for CLIFF).',
    default="/home/nlp/ariecattan/summarization/factuality/data/tofueval/mediasum_chosen.jsonl"
  )
  parser.add_argument(
    "--model_name_or_path", 
    type=str, 
    help="model name on HF on path",
    default="cattana/flan-t5-xl-qasem-joint-tokenized"
  )
  parser.add_argument(
    "--spacy_lang",
    type=str,
    help="name of spacy model",
    default="en_core_web_lg"
  )
  parser.add_argument(
    "--output_dir",
    type=str,
    help="directory to save json files"
  )
  parser.add_argument(
    "--pretokenized",
    action="store_true",
    help="whether the summary is already pretokenized"
  )
  args = parser.parse_args()
  
  # load models
  nlp = spacy.load(args.spacy_lang)
  parser = QasemParser.from_pretrained(args.model_name_or_path, spacy_lang=args.spacy_lang)

  # load data
  with jsonlines.open(args.data_path, "r") as f:
    data = [x for x in f]

  df = pd.DataFrame(data)
  
  df["datasource"] = "mediasum"
  df["origin"] = df["datasource"] 
  df["dataset"] = "tofueval"
  df["id"] = df["index"]
  # df["id"] = df.apply(lambda row: row["doc_id"] + "_" + row["topic"], axis=1)
  df["label"] = df["summary_label"].apply(lambda x: 1 if x == True else 0)

  # run spacy on source 
  spacy_docs_source = list(tqdm(nlp.pipe(df["article"]), 
                                desc='Running spacy on source', 
                                total=len(df)))
  df["spacy_source"] = spacy_docs_source

  df["spacy_summary"] = df["summary"].apply(get_spacy_summary)
    
  # split summary into sentences and create a dataframe at the sentence level
  df["sentences"] = df["spacy_summary"].apply(lambda x: [sent for sent in x.sents])
  df_sentences = df.explode("sentences")
  print(f'num sentences: {len(df_sentences)}')

  # add start sentence token to each sentence, in order to get summary-level token ids
  df_sentences["start_sentence_token"] = df_sentences["sentences"].apply(lambda x: x.start) 
  # this is a hack for re-running spacy on tokenized sentences, while reseting the index
  df_sentences["input_for_qasem"] = df_sentences["sentences"].apply(lambda sent: [token.text for token in sent]) 
  
  # run qasem parser 
  frames = parser(df_sentences["input_for_qasem"].tolist())
  df_sentences["qa_frames"] = frames   

  # create json file for each summary
  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
  
  
  all_data = {}
  for summary_id, summary_sentences in tqdm(df_sentences.groupby("id"),
                                            total=len(df),
                                            desc="Creating loc-unfaith input files"):
    loc_unfaith = LocUnfaith(summary_id, summary_sentences.reset_index()) # reset index to reset sentence id 
    json_input_file = loc_unfaith.export_summary_data()
    path = f"{args.output_dir}/{summary_id}.json"
    all_data[summary_id] = json_input_file
    with open(path, "w") as f:
      json.dump(json_input_file, f, indent=4)