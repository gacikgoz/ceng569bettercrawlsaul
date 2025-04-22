import pyterrier as pt
import ir_datasets
import pandas as pd


pt.init()

# Load FIQA dataset from ir_datasets
dataset = ir_datasets.load("beir/fiqa")

# Convert to DataFrames
docs_df = pd.DataFrame([{"docno": d.doc_id, "text": d.text} for d in dataset.docs_iter()])
queries_df = pd.DataFrame([{"qid": q.query_id, "query": q.text} for q in dataset.queries_iter()])
qrels_df = pd.DataFrame([{"qid": q.query_id, "docno": q.doc_id, "label": q.relevance} for q in dataset.qrels_iter()])

# Indexing
index_path = "./fiqa_index"
indexer = pt.IterDictIndexer(index_path, overwrite=True)
index_ref = indexer.index(({"docno": row["docno"], "text": row["text"]} for _, row in docs_df.iterrows()))

# Retrieval with BM25
bm25 = pt.BatchRetrieve(index_ref, wmodel="BM25")

# Run retrieval
results = bm25.transform(queries_df)

# Evaluate
eval = pt.Utils.evaluate(results, qrels_df, metrics=["map", "ndcg", "P@10"])
print(eval)