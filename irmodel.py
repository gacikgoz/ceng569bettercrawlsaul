import os, re, json, pickle
import ir_datasets, pyterrier as pt
import pandas as pd, numpy as np
from tqdm import tqdm
import faiss, torch, nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sentence_transformers import CrossEncoder, InputExample, SentenceTransformer
from torch.utils.data import DataLoader
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer
)
from datasets import Dataset

# 0) Settings
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
BM25_PATH         = "./fiqa_bm25"
FAISS_PATH        = "./fiqa_dense.index"
DOCNOS_PATH       = "./fiqa_docnos.pkl"
CE_PATH           = "./fiqa-cross-encoder"
QA_PATH           = "./fiqa-qa-model"
QA_TRAIN_FILE     = "./fiqa_train_qa.json"
QA_DEV_FILE       = "./fiqa_dev_qa.json"
MAX_LEN           = 512

# 1) Init
pt.init()
nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))
STEMMER   = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    toks = [t for t in text.split() if t not in STOPWORDS and len(t) > 1]
    return " ".join(STEMMER.stem(t) for t in toks)

def preprocess_query(q):
    q = re.sub(r"[()\"?'/:]", " ", q).strip().lower()
    return preprocess_text(q)

# 2) Build FIQA→SQuAD JSON
def build_qa_json(split, out):
    ds = ir_datasets.load(f"beir/fiqa/{split}")
    qmap = {q.query_id: q.text for q in ds.queries_iter()}
    store= ds.docs_store()
    qrels={}
    for r in ds.qrels_iter():
        qrels.setdefault(r.query_id, []).append(r)
    data=[]
    for qid, rels in tqdm(qrels.items(), desc=f"QAJSON {split}"):
        paras=[]
        for r in rels:
            ctx = store.get(r.doc_id).text
            ans = getattr(r, "answer", "") or ""
            start= ctx.lower().find(ans.lower()) if ans else 0
            qa  = {"id":f"{qid}-{r.doc_id}", "question":qmap[qid],
                   "answers":[{"text":ans,"answer_start":start}]}
            paras.append({"context":ctx,"qas":[qa]})
        data.append({"title":f"FIQA-{split}","paragraphs":paras})
    with open(out,"w") as f:
        json.dump({"version":"FIQA-QA","data":data},f)

if not os.path.exists(QA_TRAIN_FILE):
    build_qa_json("train", QA_TRAIN_FILE)
if not os.path.exists(QA_DEV_FILE):
    build_qa_json("dev", QA_DEV_FILE)

# 3) Train Cross-Encoder
def train_ce():
    ds= ir_datasets.load("beir/fiqa/train")
    qmap={q.query_id:q.text for q in ds.queries_iter()}
    store= ds.docs_store()
    exs=[]
    for r in tqdm(ds.qrels_iter(), desc="CE examples"):
        exs.append(InputExample(
            texts=[qmap[r.query_id], store.get(r.doc_id).text],
            label=float(r.relevance)
        ))
    loader=DataLoader(exs, shuffle=True, batch_size=16)
    ce=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2",
                    num_labels=1, max_length=MAX_LEN, device=DEVICE)
    ce.fit(train_dataloader=loader, epochs=5, output_path=CE_PATH, warmup_steps=100, optimizer_params={"lr":1e-5})
    ce.model.save_pretrained(CE_PATH)
    ce.tokenizer.save_pretrained(CE_PATH)

if not os.path.isdir(CE_PATH):
    train_ce()

# 4) Train QA model
def train_qa():
    # load JSON into flat records
    def load_rec(file):
        raw=json.load(open(file))["data"]
        recs=[]
        for art in raw:
            for para in art["paragraphs"]:
                ctx=para["context"]
                for qa in para["qas"]:
                    recs.append({
                        "question":qa["question"],
                        "context":ctx,
                        "answers":qa["answers"]
                    })
        return recs
    train_ds=Dataset.from_list(load_rec(QA_TRAIN_FILE))
    dev_ds  =Dataset.from_list(load_rec(QA_DEV_FILE))
    tok     = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
    model   = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")

    def prep(batch):
        t = tok(batch["question"], batch["context"],
                truncation="only_second", max_length=MAX_LEN,
                padding="max_length", return_offsets_mapping=True)

        off = t.pop("offset_mapping")

        # RoBERTa doesn't return token_type_ids
        # simulate with zeros (everything is considered a single segment)
        tt = [[0] * len(input_ids) for input_ids in t["input_ids"]]

        starts, ends = [], []
        for i, (o, ids) in enumerate(zip(off, tt)):
            ans = batch["answers"][i][0]
            sc, ec = ans["answer_start"], ans["answer_start"] + len(ans["text"])

            # assume entire sequence is one segment, simulate context segment detection
            ts, te = 0, len(ids) - 1
            sp, ep = 0, 0
            if o[ts][0] <= sc and o[te][1] >= ec:
                while ts < len(o) and o[ts][0] <= sc:
                    ts += 1
                sp = ts - 1
                while te >= 0 and o[te][1] >= ec:
                    te -= 1
                ep = te + 1
            starts.append(sp)
            ends.append(ep)
        t["start_positions"] = starts
        t["end_positions"] = ends
        return t
    train_tok=train_ds.map(prep, batched=True, remove_columns=["question","context","answers"], desc="Tok QA train")
    dev_tok  =dev_ds.map(prep, batched=True, remove_columns=["question","context","answers"], desc="Tok QA dev")
    args = TrainingArguments(
        output_dir=QA_PATH,
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        metric_for_best_model="f1",
        logging_steps=100
    )

    trainer=Trainer(model=model, args=args,
                    train_dataset=train_tok, eval_dataset=dev_tok,
                    tokenizer=tok)
    trainer.train()
    model.save_pretrained(QA_PATH)
    tok.save_pretrained(QA_PATH)

if not os.path.isdir(QA_PATH):
    train_qa()

# 5) Load models & pipelines
reranker = CrossEncoder(CE_PATH, num_labels=1, max_length=MAX_LEN, device=DEVICE)
qa_pipe  = pipeline("question-answering", model=QA_PATH, tokenizer=QA_PATH,
                   device=0 if torch.cuda.is_available() else -1)
embed    = SentenceTransformer("paraphrase-MiniLM-L6-v2", device=DEVICE)
emd_dim  = embed.get_sentence_embedding_dimension()

# 6) Build indexes & load dev set
ds_dev     = ir_datasets.load("beir/fiqa/dev")
queries_df = pd.DataFrame([{"qid":q.query_id, "query":preprocess_query(q.text)}
                          for q in ds_dev.queries_iter()])
qrels_df   = pd.DataFrame([{"qid":r.query_id, "docno":r.doc_id, "label":r.relevance}
                          for r in ds_dev.qrels_iter()])
store      = ds_dev.docs_store()

if not os.path.isdir(BM25_PATH):
    idx=pt.IterDictIndexer(BM25_PATH,meta=["docno"],fields=["text"],overwrite=True)
    idx.index(({"docno":d.doc_id,"text":preprocess_text(d.text)} for d in ds_dev.docs_iter()))
bm25=pt.BatchRetrieve(BM25_PATH, wmodel="BM25")

if not (os.path.exists(FAISS_PATH) and os.path.exists(DOCNOS_PATH)):
    texts,docs=[],[]
    for d in tqdm(ds_dev.docs_iter(), total=ds_dev.docs_count(), desc="Build FAISS"):
        texts.append(preprocess_text(d.text)); docs.append(d.doc_id)
    embs=embed.encode(texts, batch_size=64, convert_to_tensor=True).cpu().numpy()
    faiss.normalize_L2(embs)
    ix=faiss.IndexFlatIP(emd_dim); ix.add(embs)
    faiss.write_index(ix, FAISS_PATH)
    pickle.dump(docs, open(DOCNOS_PATH,"wb"))
faiss_ix=faiss.read_index(FAISS_PATH)

# 7) Retrieval + rerank
def retrieve_and_rerank(qid,query,top_k=100,alpha=0.5,rerank_k=20,beta=0.7):
    bm25_df=bm25.search(query).head(top_k)[["docno","score"]].rename(columns={"score":"bm25"})
    if bm25_df.empty:
        return pd.DataFrame([],columns=["qid","docno","score"])
    qn=preprocess_text(query)
    with torch.no_grad():
        q_emb=embed.encode([qn],convert_to_tensor=True).cpu().numpy()
        docs=[preprocess_text(store.get(d).text) for d in bm25_df.docno]
        p_emb=embed.encode(docs,convert_to_tensor=True).cpu().numpy()
    faiss.normalize_L2(q_emb); faiss.normalize_L2(p_emb)
    dscores=(p_emb@q_emb.T).ravel()
    df2=pd.DataFrame({"docno":bm25_df.docno.values,"dense":dscores})
    m=bm25_df.merge(df2,on="docno").fillna(0)
    m["bm_n"]=(m.bm25-m.bm25.min())/(m.bm25.max()-m.bm25.min()+1e-6)
    m["dn"]=(m.dense-m.dense.min())/(m.dense.max()-m.dense.min()+1e-6)
    m["fuse"]=alpha*m.bm_n+(1-alpha)*m.dn
    top=m.nlargest(rerank_k,"fuse").reset_index(drop=True)
    xs=reranker.predict([(qn,store.get(d).text) for d in top.docno])
    top["cx"]=(xs-xs.min())/(xs.max()-xs.min()+1e-6)
    top["score"]=beta*top.cx+(1-beta)*top.fuse
    top["qid"]=qid
    return top.nlargest(top_k,"score")[["qid","docno","score"]]

# 8) QA extraction
def answer_question(query, ranked, top_pass=5, top_ans=3):
    from nltk.tokenize import sent_tokenize
    results = []
    for d in ranked.head(top_pass).docno:
        context = store.get(d).text
        out = qa_pipe(question=query, context=context)

        # Span konumu
        ans_start, ans_end = out['start'], out['end']

        # Cevabın geçtiği snippet'ı bul
        sentences = sent_tokenize(context)
        snippet = ""
        for i, s in enumerate(sentences):
            if ans_start >= context.find(s) and ans_end <= context.find(s) + len(s):
                # Ortadaki cümle + komşu 1-2 cümleyi de al
                start_i = max(0, i-1)
                end_i = min(len(sentences), i+2)
                snippet = " ".join(sentences[start_i:end_i])
                break

        results.append({
            "docno": d,
            "answer": out["answer"],
            "score": out["score"],
            "snippet": snippet
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)[:top_ans]

# 9) Sweep & evaluate
alphas=np.arange(0.0,1.0,0.1)
results=[]
for a in tqdm(alphas,desc="Alpha sweep"):
    runs=[]
    for r in tqdm(queries_df.itertuples(),total=len(queries_df),desc="Retrieval"):
        runs.append(retrieve_and_rerank(r.qid,r.query,alpha=a))
    df_all=pd.concat(runs,ignore_index=True)
    df_all.to_csv(f"res_{a:.1f}.csv",index=False)
    m=pt.Utils.evaluate(df_all,qrels_df,metrics=["map","ndcg_cut_10","P_10","recall_10"])
    results.append({"alpha":a,**m})

pd.DataFrame(results).to_csv("alpha_metrics.csv",index=False)
print("Done")