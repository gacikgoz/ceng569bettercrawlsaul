import os
import pickle
import pyterrier as pt
import ir_datasets
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import torch
from tqdm import tqdm

pt.init()

# 1) Load FIQA dev split
dataset = ir_datasets.load("beir/fiqa/dev")


# 2) Build queries & qrels
def preprocess_query(q: str) -> str:
    return (q.replace("(", "")
            .replace(")", "")
            .replace('"', "")
            .replace("?", "")
            .replace("'", "")
            .replace("/", " ")
            .replace(":", " ")
            .strip()
            .lower())


queries = [
    {"qid": q.query_id, "query": preprocess_query(q.text)}
    for q in dataset.queries_iter()
    if preprocess_query(q.text)
]
queries_df = pd.DataFrame(queries)

qrels = [
    {"qid": r.query_id, "docno": r.doc_id, "label": r.relevance}
    for r in dataset.qrels_iter()
]
qrels_df = pd.DataFrame(qrels)

# 3) BM25 index with fields enabled
bm25_index_path = "./fiqa_bm25"
if os.path.isdir(bm25_index_path) and any(os.scandir(bm25_index_path)):
    print("Reusing existing BM25 index")
    index_ref = bm25_index_path
else:
    print("Building BM25 index (with text field)…")
    os.makedirs(bm25_index_path, exist_ok=True)
    indexer = pt.IterDictIndexer(
        bm25_index_path,
        meta=["docno"],
        fields=["text"],
        overwrite=True
    )


    def gen_docs():
        for d in tqdm(dataset.docs_iter(),
                      total=dataset.docs_count(),
                      desc="Indexing passages"):
            yield {"docno": d.doc_id, "text": d.text.lower()}


    index_ref = indexer.index(gen_docs())
    print("BM25 index built.")
bm25 = pt.BatchRetrieve(index_ref, wmodel="BM25")

# 4) Dense FAISS index setup
dense_index_file = "fiqa_dense.index"
mapping_file = "fiqa_docnos.pkl"
embed_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
dim = embed_model.get_sentence_embedding_dimension()

if os.path.exists(dense_index_file) and os.path.exists(mapping_file):
    print("Loading saved FAISS index and mapping…")
    cpu_index = faiss.read_index(dense_index_file)
    with open(mapping_file, "rb") as f:
        docnos = pickle.load(f)
else:
    print("Building FAISS CPU index…")
    texts, docnos = [], []
    for d in tqdm(dataset.docs_iter(),
                  total=dataset.docs_count(),
                  desc="Collecting passages"):
        texts.append(d.text.lower())
        docnos.append(d.doc_id)
    # encode all texts as tensors
    with torch.no_grad():
        text_tensors = embed_model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_tensor=True,
            device="cpu"
        )
        embs = text_tensors.cpu().numpy()
    faiss.normalize_L2(embs)
    cpu_index = faiss.IndexFlatIP(dim)
    cpu_index.add(embs)
    faiss.write_index(cpu_index, dense_index_file)
    with open(mapping_file, "wb") as f:
        pickle.dump(docnos, f)
    print("FAISS index built and saved.")
dense_index = cpu_index

# 5) Random-access store
store = dataset.docs_store()


# 6) Retrieval + on-the-fly dense re-rank
def retrieve_and_fuse(qid, query, top_k=100, alpha=0.5):
    bm25_df = bm25.search(query).head(top_k)[["docno", "score"]]
    if bm25_df.empty:
        return pd.DataFrame([], columns=["qid", "docno", "score"])
    bm25_df = bm25_df.rename(columns={"score": "bm25_score"})

    passages = [store.get(docno).text.lower() for docno in bm25_df.docno]

    # encode under torch.no_grad() as tensors
    with torch.no_grad():
        q_tensor = embed_model.encode(
            [query],
            convert_to_tensor=True,
            device="cpu"
        )
        p_tensors = embed_model.encode(
            passages,
            convert_to_tensor=True,
            device="cpu"
        )
        q_emb = q_tensor.cpu().numpy()
        p_embs = p_tensors.cpu().numpy()

    faiss.normalize_L2(q_emb)
    faiss.normalize_L2(p_embs)
    dense_scores = (p_embs @ q_emb.T).flatten()
    dense_df = pd.DataFrame({
        "docno": bm25_df.docno.values,
        "dense_score": dense_scores
    })

    merged = bm25_df.merge(dense_df, on="docno", how="outer").fillna(0)
    bmin, bmax = merged.bm25_score.min(), merged.bm25_score.max()
    dmin, dmax = merged.dense_score.min(), merged.dense_score.max()
    merged["bm25_n"] = (merged.bm25_score - bmin) / (bmax - bmin + 1e-6)
    merged["dense_n"] = (merged.dense_score - dmin) / (dmax - dmin + 1e-6)
    merged["score"] = alpha * merged.bm25_n + (1 - alpha) * merged.dense_n
    merged["qid"] = qid

    return merged[["qid", "docno", "score"]] \
        .sort_values("score", ascending=False) \
        .head(top_k)


# 10) Sweep alpha from 0.0 to 1.0 in steps of 0.1 and collect metrics + save each alpha_df
alpha_values = np.arange(0.0, 1.01, 0.1)
results = []

for alpha in tqdm(alpha_values, desc="Alpha sweep"):
    runs = []
    for row in tqdm(queries_df.itertuples(),
                    total=len(queries_df),
                    desc=f"Retrieval @ α={alpha:.1f}",
                    leave=False):
        runs.append(retrieve_and_fuse(row.qid, row.query, top_k=100, alpha=alpha))
    df_alpha = pd.concat(runs, ignore_index=True)

    # save this alpha's full results
    fname = f"results_alpha_{alpha:.1f}.csv"
    df_alpha.to_csv(fname, index=False)

    # Evaluate for this alpha
    metrics = pt.Utils.evaluate(
        df_alpha,
        qrels_df,
        metrics=["map", "ndcg@10", "P_10", "recall_10"]
    )
    results.append({
        "BM25 Weight": round(alpha, 2),
        "FAISS Weight": round(1-alpha, 2),
        "map": metrics["map"],
        "ndcg@10": metrics["ndcg@10"],
        "P_10": metrics["P_10"],
        "recall_10": metrics["recall_10"],
        "num_results": len(df_alpha)})

# 11) Save sweep summary metrics to CSV
sweep_df = pd.DataFrame(results)
sweep_df.to_csv("alpha_metrics.csv", index=False)
print("Saved alpha sweep metrics to alpha_metrics.csv and individual results_alpha_*.csv files")