import logging
import os
import pathlib
import random

import torch

from beir.beir import util, LoggingHandler
from beir.beir.datasets.data_loader import GenericDataLoader
from beir.beir.retrieval import models
from beir.beir.retrieval.evaluation import EvaluateRetrieval
from beir.beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.beir.retrieval.search.lexical import BM25Search as BM25

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#### Download nfcorpus.zip dataset and unzip the dataset
dataset = "nfcorpus"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

#### Provide the data_path where nfcorpus has been downloaded and unzipped
corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

#### Provide parameters for elastic-search
hostname = "localhost"  # localhost
index_name = "nfcorpus"  # nfcorpus
initialize = True

model = BM25(index_name=index_name, hostname=hostname, initialize=initialize)
retriever = EvaluateRetrieval(model)

#### Retrieve dense results (format of results is identical to qrels)
results = retriever.retrieve(corpus, queries)

logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, k_values=retriever.k_values)

#### Print top-k documents retrieved ####
top_k = 10

query_id, ranking_scores = random.choice(list(results.items()))
scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
logging.info("Query : %s\n" % queries[query_id])

logging.info("Start reranking")

device = "cpu"

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

logging.info("Using Device: {}".format(device))

#### Reranking top-100 docs using Dense Retriever model
model_path = "output/sentence-transformers/all-distilroberta-v1-v1-scifact"
model = DRES(models.SentenceBERT(model_path), batch_size=256,
             corpus_chunk_size=512 * 9999)
dense_retriever = EvaluateRetrieval(model, score_function="dot", k_values=[1, 3, 5, 10, 100])

#### Retrieve dense results (format of results is identical to qrels)
rerank_results = dense_retriever.rerank(corpus, queries, results, top_k=100)

#### Evaluate your retrieval using NDCG@k, MAP@K ...
ndcg, _map, recall, precision = dense_retriever.evaluate(qrels, rerank_results, retriever.k_values)

#### Print top-k documents retrieved ####
top_k = 10

query_id, ranking_scores = random.choice(list(rerank_results.items()))
scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
logging.info("Query : %s\n" % queries[query_id])

for rank in range(top_k):
    doc_id = scores_sorted[rank][0]
    # Format: Rank x: ID [Title] Body
    logging.info(
        "Rank %d: %s [%s] - %s\n" % (rank + 1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))
