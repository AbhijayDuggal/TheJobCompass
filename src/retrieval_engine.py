"""
JobCompass — Step 5: Retrieval Engine
Hybrid BM25 + semantic search + RRF fusion + cross-encoder reranking.

VRAM-aware: loads/unloads the reranker on each call so it doesn't conflict
with the embedding model. Both models must NOT be loaded simultaneously
on an 8GB GPU.

Import via:
    from retrieval_engine import RetrievalEngine
    engine = RetrievalEngine()
    jobs   = engine.find_jobs_for_candidate(candidate_dict)
    cands  = engine.find_candidates_for_job(job_dict)
"""

import gc
import os
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from opensearchpy import OpenSearch, RequestsHttpConnection
from sentence_transformers import CrossEncoder
from transformers import AutoModel, AutoTokenizer


class RetrievalEngine:
    """
    Stateful retrieval engine. Loads models on first use, unloads on demand.
    Designed for 8GB GPU — does NOT keep both embed + reranker in VRAM simultaneously.
    """

    def __init__(
        self,
        os_host          : str = "localhost",
        os_port          : int = 9200,
        candidates_alias : str = "candidates",
        jobs_alias       : str = "jobs",
        embed_model_name : str = "D:\huggingface_cache\models--Qwen--Qwen3-Embedding-0.6B\snapshots\97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        reranker_name    : str = "D:\huggingface_cache\models--BAAI--bge-reranker-v2-m3\snapshots\953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        hf_home          : str = "D:/huggingface_cache",
        bm25_top_k       : int = 100,
        semantic_top_k   : int = 100,
        rrf_k            : int = 60,
        rerank_top_n     : int = 10,
        embed_dim        : int = 1024,
        max_seq_len      : int = 512,
    ):
        os.environ["HF_HOME"] = hf_home
        self.candidates_alias  = candidates_alias
        self.jobs_alias        = jobs_alias
        self.bm25_top_k        = bm25_top_k
        self.semantic_top_k    = semantic_top_k
        self.rrf_k             = rrf_k
        self.rerank_top_n      = rerank_top_n
        self.embed_dim         = embed_dim
        self.max_seq_len       = max_seq_len
        self.embed_model_name  = embed_model_name
        self.reranker_name     = reranker_name
        self.hf_home           = hf_home
        self.device            = "cuda" if torch.cuda.is_available() else "cpu"

        # Models loaded lazily
        self._embed_tokenizer  = None
        self._embed_model      = None
        self._reranker         = None

        self.client = OpenSearch(
            hosts            = [{"host": os_host, "port": os_port}],
            http_compress    = True,
            use_ssl          = False,
            verify_certs     = False,
            ssl_show_warn    = False,
            connection_class = RequestsHttpConnection,
            timeout          = 30,
            max_retries      = 3,
            retry_on_timeout = True,
        )

    # ── Model management ──────────────────────────────────────────────────────

    def _load_embed_model(self, progress_cb: Optional[Callable] = None):
        if self._embed_model is not None:
            return
        # Unload reranker first if present
        self._unload_reranker()

        if progress_cb: progress_cb("Loading embedding model...")
        self._embed_tokenizer = AutoTokenizer.from_pretrained(
            self.embed_model_name, cache_dir=self.hf_home
        )
        self._embed_model = AutoModel.from_pretrained(
            self.embed_model_name,
            cache_dir   = self.hf_home,
            torch_dtype = torch.float16,
            device_map  = "auto",
        ).eval()

    def _unload_embed_model(self):
        if self._embed_model is not None:
            del self._embed_model; self._embed_model = None
        if self._embed_tokenizer is not None:
            del self._embed_tokenizer; self._embed_tokenizer = None
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

    def _load_reranker(self, progress_cb: Optional[Callable] = None):
        if self._reranker is not None:
            return
        # Unload embed model first if present
        self._unload_embed_model()

        if progress_cb: progress_cb("Loading reranker model...")
        self._reranker = CrossEncoder(
            self.reranker_name,
            max_length   = 512,
            device       = self.device,
            cache_folder = self.hf_home,
        )

    def _unload_reranker(self):
        if self._reranker is not None:
            del self._reranker; self._reranker = None
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

    def unload_all(self):
        """Free all VRAM. Call when the engine is idle."""
        self._unload_embed_model()
        self._unload_reranker()

    # ── Encoding ──────────────────────────────────────────────────────────────

    def _last_token_pool(self, last_hidden, attention_mask):
        if (attention_mask[:, -1].sum() == attention_mask.shape[0]):
            return last_hidden[:, -1]
        seq_lens = attention_mask.sum(dim=1) - 1
        return last_hidden[
            torch.arange(last_hidden.shape[0], device=last_hidden.device), seq_lens
        ]

    def encode_query(self, text: str, instruction: str, progress_cb: Optional[Callable] = None) -> np.ndarray:
        self._load_embed_model(progress_cb)
        formatted = f"Instruct: {instruction}\nQuery: {text.strip()}"
        inputs = self._embed_tokenizer(
            [formatted], max_length=self.max_seq_len,
            padding=True, truncation=True, return_tensors="pt",
        ).to(self._embed_model.device)
        with torch.no_grad():
            out = self._embed_model(**inputs)
        vec = self._last_token_pool(out.last_hidden_state, inputs["attention_mask"])
        vec = F.normalize(vec, p=2, dim=1)
        return vec.cpu().float().numpy()[0]

    # ── Search ────────────────────────────────────────────────────────────────

    def _bm25_search(self, alias, body, filters, top_k):
        query = body["query"]
        if filters:
            query = {"bool": {"must": [query], "filter": filters}}
        resp = self.client.search(index=alias, body={"query": query, "size": top_k})
        return [{"id": h["_id"], "score": h["_score"], "source": h["_source"]}
                for h in resp["hits"]["hits"]]

    def _semantic_search(self, alias, embed_text, instruction, filters, top_k, progress_cb=None):
        vec = self.encode_query(embed_text, instruction, progress_cb)
        knn = {"description_vector": {"vector": vec.tolist(), "k": top_k}}
        if filters:
            knn["description_vector"]["filter"] = {"bool": {"filter": filters}}
        resp = self.client.search(index=alias, body={"size": top_k, "query": {"knn": knn}})
        return [{"id": h["_id"], "score": h["_score"], "source": h["_source"]}
                for h in resp["hits"]["hits"]]

    def _rrf(self, bm25, semantic):
        scores, bm25_r, sem_r, sources = {}, {}, {}, {}
        for rank, hit in enumerate(bm25, 1):
            d = hit["id"]
            scores[d] = scores.get(d, 0) + 1/(self.rrf_k + rank)
            bm25_r[d] = rank; sources[d] = hit["source"]
        for rank, hit in enumerate(semantic, 1):
            d = hit["id"]
            scores[d] = scores.get(d, 0) + 1/(self.rrf_k + rank)
            sem_r[d]  = rank
            if d not in sources: sources[d] = hit["source"]
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:100]
        return [{"id": d, "rrf_score": round(s, 6), "bm25_rank": bm25_r.get(d),
                 "semantic_rank": sem_r.get(d), "source": sources[d]}
                for d, s in ranked]

    def _rerank(self, query_text, candidates, text_builder, progress_cb=None):
        if not candidates: return []
        self._load_reranker(progress_cb)
        pairs  = [(query_text, text_builder(c["source"])) for c in candidates]
        scores = self._reranker.predict(pairs, batch_size=32, show_progress_bar=False)
        for c, s in zip(candidates, scores): c["rerank_score"] = float(s)
        return sorted(candidates, key=lambda x: -x["rerank_score"])[:self.rerank_top_n]

    # ── Public API ────────────────────────────────────────────────────────────

    def find_jobs_for_candidate(
        self, candidate: dict,
        filter_seniority: Optional[list] = None,
        filter_location: Optional[str]   = None,
        filter_work_type: Optional[str]  = None,
        progress_cb: Optional[Callable]  = None,
    ) -> list[dict]:
        from retrieval_engine_helpers import (
            build_job_search_query, build_rerank_text_job, format_job_results
        )
        if progress_cb: progress_cb("Building query...")
        embed_text, bm25_body, filters = build_job_search_query(
            candidate, filter_seniority, filter_location, filter_work_type
        )
        filters.append({"term": {"is_active": True}})

        if progress_cb: progress_cb("Running BM25 search...")
        bm25 = self._bm25_search(self.jobs_alias, bm25_body, filters, self.bm25_top_k)

        if progress_cb: progress_cb("Running semantic search...")
        semantic = self._semantic_search(
            self.jobs_alias, embed_text,
            "Find jobs that match this candidate profile",
            filters, self.semantic_top_k, progress_cb
        )

        if progress_cb: progress_cb(f"BM25={len(bm25)} hits | Semantic={len(semantic)} hits. Fusing with RRF...")
        fused = self._rrf(bm25, semantic)

        if progress_cb: progress_cb(f"Reranking top {len(fused)} fused results...")
        reranked = self._rerank(embed_text, fused, build_rerank_text_job, progress_cb)

        if progress_cb: progress_cb(f"Done. Returning top {len(reranked)} jobs.")
        self.unload_all()
        return format_job_results(reranked, candidate)

    def find_candidates_for_job(
        self, job: dict,
        filter_seniority: Optional[list] = None,
        filter_location: Optional[str]   = None,
        progress_cb: Optional[Callable]  = None,
    ) -> list[dict]:
        from retrieval_engine_helpers import (
            build_candidate_search_query, build_rerank_text_candidate, format_candidate_results
        )
        if progress_cb: progress_cb("Building query...")
        embed_text, bm25_body, filters = build_candidate_search_query(
            job, filter_seniority, filter_location
        )

        if progress_cb: progress_cb("Running BM25 search...")
        bm25 = self._bm25_search(self.candidates_alias, bm25_body, filters, self.bm25_top_k)

        if progress_cb: progress_cb("Running semantic search...")
        semantic = self._semantic_search(
            self.candidates_alias, embed_text,
            "Find candidates that match this job description",
            filters, self.semantic_top_k, progress_cb
        )

        if progress_cb: progress_cb(f"BM25={len(bm25)} hits | Semantic={len(semantic)} hits. Fusing with RRF...")
        fused = self._rrf(bm25, semantic)

        if progress_cb: progress_cb(f"Reranking top {len(fused)} fused results...")
        reranked = self._rerank(embed_text, fused, build_rerank_text_candidate, progress_cb)

        if progress_cb: progress_cb(f"Done. Returning top {len(reranked)} candidates.")
        self.unload_all()
        return format_candidate_results(reranked, job)
