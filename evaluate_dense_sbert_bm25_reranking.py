import logging
import os
import pathlib

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
results = dense_retriever.retrieve(corpus, queries)

#### Evaluate your retrieval using NDCG@k, MAP@K ...
ndcg, _map, recall, precision = dense_retriever.evaluate(qrels, results, dense_retriever.k_values)

#### Provide parameters for elastic-search
hostname = "localhost"  # localhost
index_name = "nfcorpus"  # nfcorpus
initialize = True

model = BM25(index_name=index_name, hostname=hostname, initialize=initialize)
retriever = EvaluateRetrieval(model)

#### Retrieve dense results (format of results is identical to qrels)
reranked_results = retriever.rerank(corpus, queries, results, top_k=100)

logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
ndcg, _map, recall, precision = retriever.evaluate(qrels, reranked_results, k_values=dense_retriever.k_values)
