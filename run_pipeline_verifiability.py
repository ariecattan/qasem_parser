import ast 
import re 
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
from difflib import SequenceMatcher


parser = argparse.ArgumentParser()
parser.add_argument("--response_annotation", type=str, default="/home/nlp/ariecattan/summarization/factuality/qasem_parser/data/evaluating-verifiability-in-generative-search-engines/human_evaluation_annotations.jsonl")
parser.add_argument("--source_response_annotation_dir", type=str, default="/home/nlp/ariecattan/summarization/factuality/qasem_parser/data/evaluating-verifiability-in-generative-search-engines/verifiability_judgments")
parser.add_argument("--output_dir", type=str, default="/home/nlp/ariecattan/summarization/factuality/qasem_parser/data/evaluating-verifiability-in-generative-search-engines/loc_unfaith_files")
args = parser.parse_args()

IMPORTANT_POS = ['ADJ', 'ADV', 'NOUN', 'NUM', 'PROPN', 'VERB', 'PRON']
STR_RESEMBLANCE_THR = 0.8

def flatten_list(lst):
  return list(chain.from_iterable(lst))

def remove_citations(text):
  return re.sub(r'\[\d+\s*\]', '', text)



class LocUnfaith:
  def is_aligned(self, t1, t2):
    return SequenceMatcher(None, t1.lower(), t2.lower()).ratio() > STR_RESEMBLANCE_THR

  def _get_source_alignment(self, source_tokens, span):
    aligned_tokens = set()
    
    all_source_aligned_tokens = set([i for i, src_tok in enumerate(source_tokens) 
                                 if any([self.is_aligned(src_tok.lemma_, sum_tok) for sum_tok in span])])
    for src_tok in all_source_aligned_tokens:
      # either the source token is an important POS or there is an aligned token in a window of 2 with an important token
      if source_tokens[src_tok].pos_ in IMPORTANT_POS or \
          (src_tok+1 in all_source_aligned_tokens and source_tokens[src_tok+1].pos_ in IMPORTANT_POS) or \
          (src_tok-1 in all_source_aligned_tokens and source_tokens[src_tok-1].pos_ in IMPORTANT_POS):
        aligned_tokens.add(src_tok)

    return list(aligned_tokens)
    
  def _extract_qas_from_sentence(self, sentence_id, row):
    qas = []
    for predicate_frame in row["qa_frames"]:
      # update start and end tokens of predicate at the summary level
      predicate_token_index = predicate_frame.predicate.index + row["start_sentence_token"] 

      for frame in predicate_frame.arguments:
        # remove non-informative POS in the answer 
        # this enables to merge nodes in the graph (e.g., in Paris vs. Paris)
        # and show only the clean span in the first annotation step
        clean_start_answer = frame.start_token 
        while clean_start_answer <= frame.end_token and row["sentences"][clean_start_answer].pos_ not in IMPORTANT_POS:
          clean_start_answer += 1
        if clean_start_answer > frame.end_token:
          print(f"The answer {frame.text} is not a valid answer")
          continue
        
        # update start and end tokens of answers at the summary level
        clean_start_answer += row["start_sentence_token"]
        answer_start_token_index = frame.start_token + row["start_sentence_token"] 
        answer_end_token_index = frame.end_token + row["start_sentence_token"]

        # get source alignment to the answer
        lemma_span = [x.lemma_ for x in row["sentences"][frame.start_token:frame.end_token]]
        answer_source_alignment = self._get_source_alignment(row["spacy_source"], lemma_span)

        qas.append({
          "sentId": sentence_id,
          "predicateId": f'{predicate_token_index}-{predicate_token_index + 1}',
          "predicate": predicate_frame.predicate.text,
          "predicatePos": predicate_frame.predicate.pos,
          "question": frame.question,
          "answer": frame.text,
          "answerStartToken": [answer_start_token_index],
          "answerEndToken": [answer_end_token_index],
          "answerId": f'{answer_start_token_index}-{answer_end_token_index}',
          "cleanAnswerId": f'{clean_start_answer}-{answer_end_token_index}',
          "verbTokenId": frame.verb_token_id,
          "sourceIds": answer_source_alignment
        })

    return qas     
  
  
  def extract_qas_from_summary(self, sentences):
    qas = flatten_list(
      [self._extract_qas_from_sentence(i, row) for i, row in sentences.iterrows()]
    )
    for i, _ in enumerate(qas):
      qas[i]["questionId"] = i 
    return qas
  

  def extract_all_spans(self, sentences, qas):
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
        answer_id = qa["cleanAnswerId"] # use cleanAnswerId to merge different answers to the same span
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
    for i, span in enumerate(spans):
      spans[i]["id"] = i
      start, end = span["start"], span["end"]
      span_text = [x.lemma_ for x in sentences.iloc[0]["spacy_summary"][start:end]]
      source_alignment = self._get_source_alignment(sentences.iloc[0]["spacy_source"], span_text)
      spans[i]["sourceIds"] = source_alignment
      
    return spans 
  

  def get_source_tokens(self, sentences):
    return [[{
      "id": i,
      "text": token.text,
      "lemma": token.lemma_
    } for i, token in enumerate(source)] for source in sentences["spacy_source"]]
    
    
  def get_summary_tokens(self, sentences, spans):
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

    labels = [None] * len(sentences.iloc[0]["spacy_summary"])
    if "labels" in sentences.columns: # in CLIFF
      labels = sentences.iloc[0]["labels"]

    
    for sent_id, sent in enumerate(sentences.iloc[0]["spacy_summary"].sents):
      for token in sent:
        tokens.append({
          "id": token.i,
          "sent_id": sent_id,
          "text": token.text,
          "lemma": token.lemma_,
          "spans": token2span[token.i],
          "class": "token" if len(token2span[token.i]) == 0 else "mention", 
          "label": labels[token.i]
        })
    
    
    return tokens
  

  def export_source_data(self, df_sentences):
    '''
    special treatment here because each summary has its own source 
    '''
    sources = self.get_source_tokens(df_sentences)
    summaries = []
    for summary_id, summary_sentences in df_sentences.groupby("summary_id"):
      qas = self.extract_qas_from_summary(summary_sentences)
      if len(qas) == 0:
        return {}
      spans = self.extract_all_spans(summary_sentences, qas)
      summary = self.get_summary_tokens(summary_sentences, spans)
      summaries.append({
        "tokens": summary,
        "spans": spans,
        "qas": qas,
        "predicates": [i for i, span in enumerate(spans) if span["predicate"]],
        "label": summary_sentences.iloc[0]["citation_supports"],
        "summary_id": summary_id
      })
    return {
      "source": sources,
      "summaries": summaries,
      "sourceId": df_sentences.iloc[0]["source_id"],
      "datasource": df_sentences.iloc[0]["datasource"],
      "dataset": "verifiability",
    }

 

if __name__=="__main__":
  all_df = pd.DataFrame()
  for sp in ["train", "dev", "test"]:
    df_split = pd.read_json(os.path.join(args.source_response_annotation_dir, f"verifiability_judgments_{sp}.jsonl"), lines=True)
    df_split["split"] = sp 
    all_df = pd.concat([all_df, df_split])
  
  # dataframe with query, url and source text 
  df_sources = all_df[["query", "statement", "source_url", "source_text", "split"]].drop_duplicates()

  # dataframe with human annotation for each response
  df = pd.read_json(args.response_annotation, lines=True)


  # create an array of sentence-level annotation 
  data = []
  for i, row in df.iterrows():
    if row["citations"] is None: # filter responses without citations
      continue
    citation_dic = {x["text"]: x["link_target"] for x in row["citations"]}
    for sentence, annotation in row["annotation"]["statement_to_annotation"].items():
      if annotation["citation_annotations"] is None: 
        continue
      for citation in annotation["citation_annotations"]:
        data.append({
            "id": row["id"], # id of the query-response
            "query": row["query"], 
            "response": row["response"], # full response
            "sentence": remove_citations(sentence), # sentence with citation
            # original annotaiton whether the statement is something we should ask to filter not worthy sentences
            "citation_worthy": annotation["statement_is_verification_worthy"], 
            "citation_text": citation["citation_text"], # id of the cited source
            "citation_supports": citation["citation_supports"], # human label whether the source supports the claim
            "evidence": citation["evidence"], # human annotated evidence
            "url": citation_dic[citation["citation_text"]], # url of the citation
            "datasource": row["split"], # datasource of the query
            "system_name": row["system_name"] # name of the model (perplexity, neeva, bing_chat, you)
        })

  df_citations = pd.DataFrame(data)
  df_final = pd.merge(
    df_citations, 
    df_sources, 
    how="inner", 
    left_on=["query", "sentence", "url"],
    right_on=["query", "statement", "source_url"]
  )

  # take only instances that contains at least one citation with partial support or no support
  interesting_labels = [
    "Citation Completely Supports but Also Refutes Statement",
    "Citation Partially Supports Statement",
    "Citation Provides No Support for Statement"
  ]

  group_ids = []
  for group_id, group in df_final.groupby("id"):
    citation_labels = set(group["citation_supports"])
    if len(group) > 3 and \
      (interesting_labels[0] in citation_labels or \
       interesting_labels[1] in citation_labels or \
        interesting_labels[2] in citation_labels):
      group_ids.append(group_id)
  print(f'Number of potential topics: {len(group_ids)}')

  df_final = df_final[df_final["id"].isin(set(group_ids))].copy()

  # df_final = df_final[:20].copy()


  # load models 
  nlp = spacy.load("en_core_web_lg")
  parser = QasemParser.from_pretrained("cattana/flan-t5-xl-qasem-joint-tokenized", spacy_lang="en_core_web_lg")
  loc_unfaith = LocUnfaith()

  # run spacy on the source and the generated statement 
  df_final["spacy_source"] = list(tqdm(nlp.pipe(list(df_final["source_text"])), desc="Running spacy on source", total=len(df_final)))
  df_final["spacy_summary"] = list(tqdm(nlp.pipe(list(df_final["sentence"])), desc="Running spacy on summary", total=len(df_final)))
  df_final["summary_tokens"] = df_final["spacy_summary"].apply(lambda s: [x.text for x in s])
  df_final["sentences"] = df_final["spacy_summary"].apply(lambda x: [sent for sent in x.sents])
  df_final["index"] = list(range(len(df_final)))
  
  df_final["summary_id"] = df_final.apply(lambda row: f"{row['id']}_{row['index']}", axis=1)
  df_final["source_id"] = df_final["id"]

  

  df_sentences = df_final.explode("sentences")
  df_sentences["start_sentence_token"] = df_sentences["sentences"].apply(lambda x: x.start) 
  df_sentences["input_for_qasem"] = df_sentences["sentences"].apply(lambda sent: [token.text for token in sent]) 
    
  # run qasem parser 
  frames = parser(df_sentences["input_for_qasem"].tolist())
  df_sentences["qa_frames"] = frames  


  for source_id, sentences in tqdm(df_sentences.groupby("id"), total=df_sentences["id"].nunique()):
    loc_unfaith_file = loc_unfaith.export_source_data(sentences)
    with open(os.path.join(args.output_dir, f"{source_id}.json"), "w") as f:
      json.dump(loc_unfaith_file, f, indent=4)