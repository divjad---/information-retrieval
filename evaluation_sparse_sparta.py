import logging
import os
import pathlib
import random

import torch

from beir.beir import util, LoggingHandler
from beir.beir.datasets.data_loader import GenericDataLoader
from beir.beir.retrieval import models
from beir.beir.retrieval.evaluation import EvaluateRetrieval
from beir.beir.retrieval.search.sparse import SparseSearch

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

dataset = "nfcorpus"

out_dir = "datasets"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), out_dir)
data_path = util.download_and_unzip(url, out_dir)

corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

device = "cpu"

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

#### Sparse Retrieval using SPARTA ####
model_path = "output/sentence-transformers/all-distilroberta-v1-v2-scifact-bm25-hard-negs"
#model_path = "output/distilbert-base-uncased-v1-scifact"
sparse_model = SparseSearch(models.SPARTA(model_path, device=device), batch_size=128)
# model = DRES(models.SentenceBERT(model_path), batch_size=128, corpus_chunk_size=512*9999)
retriever = EvaluateRetrieval(sparse_model)

#### Retrieve dense results (format of results is identical to qrels)
results = retriever.retrieve(corpus, queries)

#### Evaluate your retrieval using NDCG@k, MAP@K ...

logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

#### Print top-k documents retrieved ####
top_k = 10

query_id, ranking_scores = random.choice(list(results.items()))
scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
logging.info("Query : %s\n" % queries[query_id])

for rank in range(top_k):
    doc_id = scores_sorted[rank][0]
    # Format: Rank x: ID [Title] Body
    logging.info(
        "Rank %d: %s [%s] - %s\n" % (rank + 1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))
