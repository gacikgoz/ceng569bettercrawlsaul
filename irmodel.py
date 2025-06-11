import csv
import json
import os
import pickle
import re
from collections import Counter

import evaluate
import faiss
import ir_datasets
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import pyterrier as pt
import torch
from datasets import Dataset
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sentence_transformers import CrossEncoder, InputExample, SentenceTransformer
from sklearn.metrics import precision_recall_curve, auc
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer
)

# 0) Settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BM25_PATH = "./fiqa_bm25"
FAISS_PATH = "./fiqa_dense.index"
DOCNOS_PATH = "./fiqa_docnos.pkl"
CE_PATH = "./fiqa-cross-encoder"
QA_PATH = "./fiqa-qa-model"
QA_TRAIN_FILE = "./fiqa_train_qa.json"
QA_DEV_FILE = "./fiqa_dev_qa.json"
MAX_LEN = 512

# 1) Init
pt.init()
nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    toks = [t for t in text.split() if t not in STOPWORDS and len(t) > 1]
    return " ".join(STEMMER.stem(t) for t in toks)


def preprocess_query(q):
    q = re.sub(r"[()\"?'/:]", " ", q).strip().lower()
    return preprocess_text(q)


# 3) Train Cross-Encoder
def train_ce(negatives_per_query=3):
    ds = ir_datasets.load("beir/fiqa/train")
    qmap = {q.query_id: q.text for q in ds.queries_iter()}
    store = ds.docs_store()
    all_doc_ids = [d.doc_id for d in ds.docs_iter()]

    # Pozitif qrel'leri topla
    pos_set = set()
    for r in ds.qrels_iter():
        pos_set.add((r.query_id, r.doc_id))

    examples = []

    # Pozitif örnekler
    for qid, did in pos_set:
        query = qmap[qid]
        doc = store.get(did).text
        examples.append(InputExample(texts=[query, doc], label=1.0))

    # Negatif örnekler
    import random
    for qid in qmap:
        used_docs = [did for (qid2, did) in pos_set if qid2 == qid]
        if not used_docs:
            continue
        negatives = random.sample(
            [d for d in all_doc_ids if d not in used_docs],
            min(negatives_per_query, len(all_doc_ids))
        )
        for did in negatives:
            doc = store.get(did).text
            examples.append(InputExample(texts=[qmap[qid], doc], label=0.0))

    print(f"[CE Train] {len(examples)} total examples ({len(pos_set)} pos + ~{len(examples) - len(pos_set)} neg)")

    loader = DataLoader(examples, shuffle=True, batch_size=16)

    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2",
                      num_labels=1,
                      max_length=MAX_LEN,
                      device=DEVICE)

    ce.fit(train_dataloader=loader,
           epochs=5,
           output_path=CE_PATH,
           warmup_steps=100,
           optimizer_params={"lr": 1e-5})

    ce.model.save_pretrained(CE_PATH)
    ce.tokenizer.save_pretrained(CE_PATH)


if not os.path.isdir(CE_PATH):
    train_ce()

# 5) Load models & pipelines
reranker = CrossEncoder(CE_PATH, num_labels=1, max_length=MAX_LEN, device=DEVICE)

embed = SentenceTransformer("paraphrase-MiniLM-L6-v2", device=DEVICE)
emd_dim = embed.get_sentence_embedding_dimension()

# 6) Build indexes & load dev set
ds_dev = ir_datasets.load("beir/fiqa/dev")
queries_df = pd.DataFrame([{"qid": q.query_id, "query": preprocess_query(q.text)}
                           for q in ds_dev.queries_iter()])
qrels_df = pd.DataFrame([{"qid": r.query_id, "docno": r.doc_id, "label": r.relevance}
                         for r in ds_dev.qrels_iter()])
store = ds_dev.docs_store()

if not os.path.isdir(BM25_PATH):
    idx = pt.IterDictIndexer(BM25_PATH, meta=["docno"], fields=["text"], overwrite=True)
    idx.index(({"docno": d.doc_id, "text": preprocess_text(d.text)} for d in ds_dev.docs_iter()))

bm25 = pt.BatchRetrieve(BM25_PATH, wmodel="BM25")

if not (os.path.exists(FAISS_PATH) and os.path.exists(DOCNOS_PATH)):
    texts, docs = [], []
    for d in tqdm(ds_dev.docs_iter(), total=ds_dev.docs_count(), desc="Build FAISS"):
        texts.append(preprocess_text(d.text));
        docs.append(d.doc_id)
    embs = embed.encode(texts, batch_size=64, convert_to_tensor=True).cpu().numpy()
    faiss.normalize_L2(embs)
    ix = faiss.IndexFlatIP(emd_dim);
    ix.add(embs)
    faiss.write_index(ix, FAISS_PATH)
    pickle.dump(docs, open(DOCNOS_PATH, "wb"))
faiss_ix = faiss.read_index(FAISS_PATH)


# 7) Retrieval + rerank
def retrieve_and_rerank(qid, query, top_k=100, alpha=0.5, rerank_k=20, beta=0.7):
    bm25_df = bm25.search(query).head(top_k)[["docno", "score"]].rename(columns={"score": "bm25"})
    if bm25_df.empty:
        return pd.DataFrame([], columns=["qid", "docno", "score"])
    qn = preprocess_text(query)
    with torch.no_grad():
        q_emb = embed.encode([qn], convert_to_tensor=True).cpu().numpy()
        docs = [preprocess_text(store.get(d).text) for d in bm25_df.docno]
        p_emb = embed.encode(docs, convert_to_tensor=True).cpu().numpy()
    faiss.normalize_L2(q_emb);
    faiss.normalize_L2(p_emb)
    dscores = (p_emb @ q_emb.T).ravel()
    df2 = pd.DataFrame({"docno": bm25_df.docno.values, "dense": dscores})
    m = bm25_df.merge(df2, on="docno").fillna(0)
    m["bm_n"] = (m.bm25 - m.bm25.min()) / (m.bm25.max() - m.bm25.min() + 1e-6)
    m["dn"] = (m.dense - m.dense.min()) / (m.dense.max() - m.dense.min() + 1e-6)
    m["fuse"] = alpha * m.bm_n + (1 - alpha) * m.dn
    top = m.nlargest(rerank_k, "fuse").reset_index(drop=True)
    xs = reranker.predict([(qn, store.get(d).text) for d in top.docno])
    if xs.max() - xs.min() < 1e-6:
        top["cx"] = 0.5  # fallback
    else:
        top["cx"] = (xs - xs.min()) / (xs.max() - xs.min() + 1e-6)
    top["score"] = beta * top.cx + (1 - beta) * top.fuse
    top["qid"] = qid
    return top.nlargest(top_k, "score")[["qid", "docno", "score"]]

qa_pipe = pipeline("question-answering",
                   model="deepset/roberta-base-squad2",
                   tokenizer="deepset/roberta-base-squad2",
                   device=0 if torch.cuda.is_available() else -1)

# 8) QA extraction
def answer_question(query, ranked, top_pass=5, top_ans=3):
    from nltk.tokenize import sent_tokenize
    results = []

    for d in ranked.head(top_pass).docno:
        context = store.get(d).text
        try:
            out = qa_pipe(question=query, context=context)
            answer = out.get("answer", "").strip()
            score = out.get("score", 0.0)
        except Exception as e:
            continue  # QA hatası varsa o dokümanı atla

        # Eğer cevap boşsa bu sonucu alma
        if not answer:
            continue

        # Cevabın geçtiği snippet'ı bul
        sentences = sent_tokenize(context)
        snippet = ""
        for i, s in enumerate(sentences):
            if answer.lower() in s.lower():
                start_i = max(0, i - 1)
                end_i = min(len(sentences), i + 2)
                snippet = " ".join(sentences[start_i:end_i])
                break

        results.append({
            "docno": d,
            "answer": answer,
            "score": score,
            "snippet": snippet
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)[:top_ans]



# 9) Sweep & evaluate
def evaluate_reranker_model(reranker, ds, output_csv="reranker_eval_results.csv", threshold=0.5):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    import csv

    qmap = {q.query_id: q.text for q in ds.queries_iter()}
    store = ds.docs_store()

    queries = []
    passages = []
    labels = []

    for r in ds.qrels_iter():
        queries.append(qmap[r.query_id])
        passages.append(store.get(r.doc_id).text)
        labels.append(1 if r.relevance > 0 else 0)

    probs = reranker.predict(list(zip(queries, passages)))
    preds = [1 if p >= threshold else 0 for p in probs]

    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)

    results = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        writer.writeheader()
        writer.writerow(results)

    print(f"[Reranker Eval] Acc: {acc:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f} | F1: {f1:.3f}")


def plot_pr_curve(qrels, run_df, title="PR Curve", save_path=None):
    y_true = []
    y_scores = []

    qrels_dict = {(row['qid'], row['docno']): row['label'] for _, row in qrels.iterrows()}

    for _, row in run_df.iterrows():
        label = qrels_dict.get((row['qid'], row['docno']), 0)
        y_true.append(label)
        y_scores.append(row['score'])

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f'PR AUC={pr_auc:.4f}')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

evaluate_reranker_model(reranker, ir_datasets.load("beir/fiqa/dev"))

alphas = np.arange(0.0, 1.1, 0.2)
betas = np.arange(0.0, 1.1, 0.2)
results = []

for a in alphas:
    for b in betas:
        runs = []
        for r in tqdm(queries_df.itertuples(), total=len(queries_df),
                      desc=f"α={a:.1f}, β={b:.1f} Retrieval"):
            runs.append(retrieve_and_rerank(r.qid, r.query, alpha=a, beta=b))

        df_all = pd.concat(runs, ignore_index=True)
        df_all.to_csv(f"res_a{a:.1f}_b{b:.1f}.csv", index=False)

        # PR Curve
        plot_pr_curve(qrels_df, df_all,
                      title=f"PR Curve α={a:.1f}, β={b:.1f}",
                      save_path=f"pr_curve_a{a:.1f}_b{b:.1f}.png")

        # Global IR metrics
        m = pt.Utils.evaluate(df_all, qrels_df,
                              metrics=["map", "ndcg_cut_10", "P_10", "P_20", "recall_10", "recall_20"])
        results.append({"alpha": a, "beta": b, **m})

# Save all results to one CSV
pd.DataFrame(results).to_csv("alpha_beta_metrics.csv", index=False)
print("Done")
