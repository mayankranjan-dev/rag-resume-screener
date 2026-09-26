"""
Retrieval hit-rate evaluation for the RAG Resume Screener.

Indexes a hardcoded mock resume, runs a fixed set of queries, and reports
the percentage of queries where at least one of the top-k retrieved chunks
contains the expected ground-truth keyword.

Usage (from the project root):
    python eval.py
"""

from app.services.ingest import chunk_text
from app.services.rag import delete_resume_chunks, index_resume_chunks, retrieve_top_chunks

TEST_RESUME_ID = "eval-test-resume"
TOP_K = 3
# Smaller than the app default (500) so this short resume yields enough
# chunks for top-3 retrieval to be a meaningful test rather than "return everything".
CHUNK_SIZE = 200
CHUNK_OVERLAP = 20

MOCK_RESUME = """Jordan Example - Senior Backend Engineer

Summary: Backend engineer with 5 years Python experience building high-throughput APIs with FastAPI and Django. Comfortable owning services end to end.

Database Work: Led PostgreSQL optimization efforts, rewriting slow queries and adding composite indexes, which cut p95 query latency by 60 percent. Also maintained Redis caching layers.

Infrastructure: Owned Kubernetes deployment pipelines for 30+ microservices using Helm charts and ArgoCD. Configured horizontal pod autoscaling and rolling updates.

Cloud: Designed AWS architectures using ECS, Lambda, S3 and RDS. Managed infrastructure as code with Terraform.

Education: B.S. in Computer Science from State University, graduated 2019. Coursework in distributed systems and databases.

Leadership: Mentored four junior engineers and ran weekly code review sessions for the platform team.
"""

# (query, expected keyword that must appear in at least one retrieved chunk)
TEST_CASES = [
    ("How many years of Python experience does the candidate have?", "5 years Python"),
    ("Has the candidate worked on database performance tuning?", "PostgreSQL optimization"),
    ("What container orchestration experience do they have?", "Kubernetes deployment"),
    ("What is the candidate's educational background?", "Computer Science"),
    ("Which cloud provider has the candidate used?", "AWS"),
]


def main() -> None:
    chunks = chunk_text(MOCK_RESUME, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    index_resume_chunks(TEST_RESUME_ID, chunks)
    print(f"Indexed {len(chunks)} chunks under '{TEST_RESUME_ID}' (top_k={TOP_K})\n")

    try:
        rows = []
        for query, keyword in TEST_CASES:
            results = retrieve_top_chunks(query, TEST_RESUME_ID, top_k=TOP_K)
            hit_ranks = [
                rank for rank, r in enumerate(results, start=1)
                if keyword.lower() in r["text"].lower()
            ]
            rows.append((query, keyword, hit_ranks[0] if hit_ranks else None))
    finally:
        # Don't leave eval data in the real store.
        delete_resume_chunks(TEST_RESUME_ID)

    q_w = max(len(r[0]) for r in rows)
    k_w = max(len(r[1]) for r in rows)
    header = f"{'Query':<{q_w}} | {'Expected Keyword':<{k_w}} | Result"
    print(header)
    print("-" * len(header))
    for query, keyword, rank in rows:
        result = f"HIT (rank {rank})" if rank else "MISS"
        print(f"{query:<{q_w}} | {keyword:<{k_w}} | {result}")

    hits = sum(1 for r in rows if r[2])
    print("-" * len(header))
    print(f"Retrieval Hit-Rate @ {TOP_K}: {hits}/{len(rows)} = {hits / len(rows):.0%}")


if __name__ == "__main__":
    main()
