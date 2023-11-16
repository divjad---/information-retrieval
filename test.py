import torch
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder, util

if not torch.cuda.is_available():
    print("Warning: No GPU found. Please add GPU to your notebook")

# We use the Bi-Encoder to encode all passages, so that we can use it with sematic search
bi_encoder = SentenceTransformer("output/all-mpnet-base-v2-v1-scifact")
#bi_encoder.max_seq_length = 128  # Truncate long passages to 128 tokens
top_k = 32  # Number of passages we want to retrieve with the bi-encoder

# The bi-encoder will retrieve 100 documents. We use a cross-encoder, to re-rank the results list to improve the quality
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


# As dataset, we use Simple English Wikipedia. Compared to the full English wikipedia, it has only
# about 170k articles. We split these articles into paragraphs and encode them with the bi-encoder

def remove_newlines(serie):
    serie = serie.replace('\n', ' ')
    serie = serie.replace('\\n', ' ')
    serie = serie.replace('  ', ' ')
    serie = serie.replace('  ', ' ')
    return serie


pdf_reader = PdfReader("./datasets/play.pdf")
# Text variable will store the pdf text
text = ""
for page in pdf_reader.pages:
    text += remove_newlines(page.extract_text())

print(len(text.split(".")))

limit = 384


def chunker(txt: str):
    chunks = []
    all_contexts = txt.split('.')
    chunk = []
    for context in all_contexts:
        chunk.append(context)
        if len(chunk) >= 3 and len('.'.join(chunk)) > limit:
            # surpassed limit so add to chunks and reset
            chunks.append('.'.join(chunk).strip() + '.')
            # add some overlap between passages
            chunk = chunk[-2:]
    # if we finish and still have a chunk, add it
    if chunk is not None:
        chunks.append('.'.join(chunk))
    return chunks


passages = chunker(text)
print("Passages:", len(passages))

# We encode all passages into our vector space. This takes about 5 minutes (depends on your GPU speed)
corpus_embeddings = bi_encoder.encode(passages, convert_to_tensor=True, show_progress_bar=True)


def search(query):
    print("Input question:", query)

    ##### Sematic Search #####
    # Encode the query using the bi-encoder and find potentially relevant passages
    question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)
    hits = hits[0]  # Get the hits for the first query

    ##### Re-Ranking #####
    # Now, score all retrieved passages with the cross_encoder
    cross_inp = [[query, passages[hit['corpus_id']]] for hit in hits]
    cross_scores = cross_encoder.predict(cross_inp)

    # Sort results by the cross-encoder scores
    for idx in range(len(cross_scores)):
        hits[idx]['cross-score'] = cross_scores[idx]

    # Output of top-5 hits from bi-encoder
    print("\n-------------------------\n")
    print("Top-3 Bi-Encoder Retrieval hits")
    hits = sorted(hits, key=lambda x: x['score'], reverse=True)
    for hit in hits[0:3]:
        print("\t{:.3f}\t{}".format(hit['score'], passages[hit['corpus_id']].replace("\n", " ")))

    # Output of top-5 hits from re-ranker
    print("\n-------------------------\n")
    print("Top-3 Cross-Encoder Re-ranker hits")
    hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
    for hit in hits[0:3]:
        print("\t{:.3f}\t{}".format(hit['cross-score'], passages[hit['corpus_id']].replace("\n", " ")))

    return [passages[hit['corpus_id']] for hit in hits]


passages = search("What are the eight play personality types described in chapter one?")
for passage in passages:
    print(passage)
    print("========================================")