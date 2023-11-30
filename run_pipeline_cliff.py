import os 
import pandas as pd 
from tqdm import tqdm 
import spacy 
from itertools import chain
import collections 
from qanom.spacy_component_nominalization_detector import *
import json 
from spacy.tokens import Doc
import jsonlines
from qasem_parser import QasemParser, QasemFrame, QasemArgument

tqdm.pandas()

def flatten_list(lst):
  return list(chain.from_iterable(lst))


class LocUnfaith:
  def __init__(self, summary_id, summary_sentences) -> None:
    self.summary_id = summary_id
    self.df_sentences = summary_sentences
    

  def _extract_qas_from_sentence(self, sentence_id, row):
    qas = []
    for predicate_frame in row["qa_frames"]:
      # update start and end tokens of predicate at the summary level
      predicate_token_index = predicate_frame.predicate.index + row["start_sentence_token"] 

      for frame in predicate_frame.arguments:
        # update start and end tokens of answers at the summary level
        answer_start_token_index = frame.start_token + row["start_sentence_token"] 
        answer_end_token_index = frame.end_token + row["start_sentence_token"] 

        qas.append({
          "sentId": sentence_id,
          "predicateId": f'{predicate_token_index}-{predicate_token_index + 1}',
          "predicate": predicate_frame.predicate.text,
          "predicatePos": predicate_frame.predicate.pos,
          "question": frame.question,
          "answer": frame.text,
          "answerStartToken": [answer_start_token_index],
          "answerEndToken": [answer_end_token_index],
          "answerId": f'{answer_start_token_index}-{answer_end_token_index}'
        })

    return qas     
  
  
  def extract_qas_from_summary(self):
    qas = flatten_list(
      [self._extract_qas_from_sentence(i, row) for i, row in self.df_sentences.iterrows()]
    )
    for i, _ in enumerate(qas):
      qas[i]["questionId"] = i 
    return qas
  

  def extract_all_spans(self, qas):
    spans = []

    '''
    predicates
    answers
    '''
    predicates = collections.defaultdict(list)
    answers = collections.defaultdict(list)
    predicate_tokens = set() # to check afterwards whether spans include predicate

    df_qas = pd.DataFrame(qas)

    # create mapping from predicate_id and answer_id to question_ids they are involved
    for predicate_id, predicate_qas in df_qas.groupby("predicateId"):
      predicate_tokens.add(int(predicate_id.split("-")[0]))
      predicates[predicate_id] = predicate_qas["questionId"].tolist()
      for i, qa in predicate_qas.iterrows():
        answer_id = qa["answerId"]
        answers[answer_id].append(qa["questionId"])

    spans = []

    # adding predicates
    for predicate_id, qa_ids in predicates.items():
      predicate_start, predicate_end = predicate_id.split("-")
      spans.append({
        "start": int(predicate_start),
        "end": int(predicate_end),
        "qaIds": qa_ids,
        "predicate": True,
        "include_predicate": True
      })

    # adding answers
    for answer_id, qa_ids in answers.items():
      answer_start, answer_end = answer_id.split("-")
      answer_start, answer_end = int(answer_start), int(answer_end)
      answer_tokens = set(range(answer_start, answer_end))
      # check if one the answer token is a predicate
      include_predicate = len(answer_tokens.intersection(predicate_tokens)) > 0 
      spans.append({
        "start": answer_start,
        "end": answer_end,
        "qaIds": qa_ids,
        "predicate": False,
        "include_predicate": include_predicate
      })

    spans = sorted(spans, key=lambda x: x["start"])
    for i, _ in enumerate(spans):
      spans[i]["id"] = i
      
    return spans 
  

  def get_source_tokens(self):
    return [{
      "id": i,
      "text": token.text,
      "lemma": token.lemma_
    } for i, token in enumerate(self.df_sentences.iloc[0]["spacy_source"])]
    
  def get_summary_tokens(self, spans):
    """
    For each token, add the span id and the corresponding class (token or mention)
    token is a standard token
    mention is a token that participates either as a predicate in an answer
    """
    tokens = []

    token2span = collections.defaultdict(list)
    for i, span in enumerate(spans):
      for token_id in range(span["start"], span["end"]):
        token2span[token_id].append(i)

    for i, token in enumerate(self.df_sentences.iloc[0]["spacy_summary"]):
      tokens.append({
        "id": i,
        "text": token.text,
        "lemma": token.lemma_,
        "spans": token2span[i],
        "class": "token" if len(token2span[i]) == 0 else "mention"
      })
    
    return tokens

  def export_summary_data(self):
    """
    Post process the output of qasem parser 
    and prepare JSON input files for loc-unfaith 
    """
    source = self.get_source_tokens()
    qas = self.extract_qas_from_summary()
    if len(qas) == 0:
      return {}
    
    spans = self.extract_all_spans(qas)
    summary = self.get_summary_tokens(spans)

    return {
      "source": source,
      "summary": summary,
      "spans": spans,
      "qas": qas,
      "datasource": self.df_sentences.iloc[0]["origin"],
      "summaryId": f'{self.df_sentences.iloc[0]["id"]}',
      "aggrefactId": self.summary_id,
      "label": int(self.df_sentences.iloc[0]["label"]),
      "dataset": str(self.df_sentences.iloc[0]["dataset"])
    }


if __name__=='__main__':
  nlp = spacy.load("en_core_web_lg")
  nlp.add_pipe("nominalization_detector", after="tagger", 
             config={"threshold": 0.75, "device": -1})
  arg_parser_path = "cattana/flan-t5-xl-qasem-joint-tokenized"
  parser = QasemParser.from_pretrained(arg_parser_path, spacy_lang="en_core_web_lg")

  # df = pd.read_csv("../AggreFact/data/aggre_fact_sota.csv")
  # df = pd.read_json("/home/nlp/ariecattan/summarization/factuality/cliff/cliff_raw.jsonl")
  with jsonlines.open("/home/nlp/ariecattan/summarization/factuality/cliff/cliff_raw.jsonl", "r") as f:
    data = [x for x in f]
  df = pd.DataFrame(data)
  df["label"] = df["labels"].apply(lambda labels: True if sum(1 for x in labels if x == "correct") == len(labels) else False) 
  df["origin"] = df["datasource"]
  df["dataset"] = "CLIFF"
  df["summary_tokens"] = df["summary"].apply(lambda s: s.split())
  

  # run spacy on source and summary 
  spacy_docs_source = list(tqdm(nlp.pipe(df["article"], disable=["nominalization_detector"]), desc='Running spacy on source', total=len(df)))
  df["spacy_source"] = spacy_docs_source
  # spacy_docs_summary = list(tqdm(nlp.pipe(df["summary"]), desc='Running spacy on summary', total=len(df)))
  # df["spacy_summary"] = spacy_docs_summary

  # run spacy on pretokenized summary to further split it into sentences
  df["spacy_inputs"] =  df["summary_tokens"].progress_apply(lambda tokens: Doc(words=tokens, vocab=nlp.vocab)) 
  docs = list(tqdm(nlp.pipe(list(df["spacy_inputs"])), desc="Running spacy on summary", total=len(df)))
  df["spacy_summary"] = docs

  # for running qasem on sentences, we first need to transform each sentence to a spacy 
  # otherwise, it makes troubles with the index of the predicate and argument
  df["sentences"] = df["spacy_summary"].apply(lambda x: [sent for sent in x.sents])
  df_sentences = df.explode("sentences")
  print(f'num sentences: {len(df_sentences)}')
  df_sentences["start_sentence_token"] = df_sentences["sentences"].apply(lambda x: x.start)
  # this is a hack for re-running spacy on tokenized sentences, while reseting the index
  df_sentences["input_for_qasem"] = df_sentences["sentences"].apply(lambda sent: [token.text for token in sent]) 
  
  # run qasem parser
  # frames = list(tqdm(parser(df_sentences["input_for_qasem"].tolist()), desc='Running qasem parser...', total=len(df_sentences)))
  
  bugs = []
  for i, sentence in tqdm(enumerate(df_sentences["input_for_qasem"])):
    try:
      predicates = parser.predicate_detector.predict([sentence])
    except:
      bugs.append(sentence)

  exit()
  '''
  
  
  '''
  frames = parser(df_sentences["input_for_qasem"].tolist())
  df_sentences["qa_frames"] = frames 

  # create json file for each summary
  if not os.path.exists("data/cliff_t5_xl"):
    os.makedirs("data/cliff_t5_xl")
  all_data = {}
  for summary_id, summary_sentences in tqdm(df_sentences.groupby("id"),
                                            total=len(df),
                                            desc="Creating loc-unfaith input files"):
    loc_unfaith = LocUnfaith(summary_id, summary_sentences)
    json_input_file = loc_unfaith.export_summary_data()
    path = f"data/cliff_t5_xl/{summary_id}.json"
    all_data[summary_id] = json_input_file
    # with open(path, "w") as f:
    #   json.dump(json_input_file, f, indent=4)