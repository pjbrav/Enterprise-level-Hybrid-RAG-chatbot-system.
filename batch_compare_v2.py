"""
Batch comparison eval script v2 — CLI args, file picker, error handling.

Usage:
    python batch_compare_v2.py                                    # defaults
    python batch_compare_v2.py --db mydb.db --questions qs.parquet # specify files
    python batch_compare_v2.py --sample 50 --provider openai       # 50 questions
    python batch_compare_v2.py --pick                              # file picker dialog
    python batch_compare_v2.py --force-hybrid                      # run both on every question
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Standard vs Hybrid RAG batch comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--db", default=os.getenv("SQLITE_DATABASE_PATH", "hybrid_rag_v22.db"),
                   help="Path to SQLite database (default: hybrid_rag_v22.db or SQLITE_DATABASE_PATH env)")
    p.add_argument("--questions", default="questionaskerbigset.parquet",
                   help="Path to questions parquet file (default: questionaskerbigset.parquet)")
    p.add_argument("--sample", type=int, default=200,
                   help="Number of questions to sample (default: 200, 0=all)")
    p.add_argument("--provider", default="openai",
                   help="LLM provider: openai, claude, gemini (default: openai)")
    p.add_argument("--output", default="comparison_results",
                   help="Output file prefix (default: comparison_results -> .csv and .md)")
    p.add_argument("--pick", action="store_true",
                   help="Open file picker dialog to select DB and questions file")
    p.add_argument("--force-hybrid", action="store_true",
                   help="Run both Standard and Hybrid on every question (eval mode)")
    p.add_argument("--no-direct-ai", action="store_true", default=True,
                   help="Skip Direct AI call (default: skipped)")
    return p.parse_args()


def pick_file(title: str, filetypes: list = None):
    """Open a file picker dialog. Returns path or None."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(title=title, filetypes=filetypes or [("All files", "*.*")])
        root.destroy()
        return path if path else None
    except ImportError:
        print("  [WARNING] tkinter not available — cannot open file picker.")
        print("           Install with: pip install tk  (or use --db and --questions args)")
        return None


def main():
    args = parse_args()

    # File picker mode
    if args.pick:
        db_path = pick_file("Select SQLite database", [("SQLite DB", "*.db"), ("All files", "*.*")])
        if not db_path:
            print("No database selected. Exiting.")
            sys.exit(1)
        questions_path = pick_file("Select questions parquet", [("Parquet", "*.parquet"), ("All files", "*.*")])
        if not questions_path:
            print("No questions file selected. Exiting.")
            sys.exit(1)
    else:
        db_path = args.db
        questions_path = args.questions

    # Validate files exist
    if not os.path.exists(db_path):
        print(f"ERROR: Database not found: {db_path}")
        print(f"  Set SQLITE_DATABASE_PATH env var or use --db /path/to/db.db")
        sys.exit(1)
    if not os.path.exists(questions_path):
        print(f"ERROR: Questions file not found: {questions_path}")
        print(f"  Use --questions /path/to/questions.parquet")
        sys.exit(1)

    # Check optional dependencies
    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas not installed. Run: pip install pandas pyarrow")
        sys.exit(1)

    # Check spaCy
    try:
        import spacy
        spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
        spacy_status = "✅"
    except ImportError:
        spacy_status = "⚠️ not installed (regex fallback)"
    except Exception:
        spacy_status = "⚠️ model not downloaded (run: python -m spacy download en_core_web_sm)"

    sample_size = args.sample if args.sample > 0 else 999999

    print(f"\n{'=' * 70}")
    print(f"  Standard vs Hybrid RAG — Batch Comparison")
    print(f"{'=' * 70}")
    print(f"  Database:   {db_path}")
    print(f"  Questions:  {questions_path}")
    print(f"  Sample:     {sample_size if sample_size < 999999 else 'all'}")
    print(f"  Provider:   {args.provider}")
    print(f"  spaCy:      {spacy_status}")
    print(f"  Mode:       {'forced hybrid (both run)' if args.force_hybrid else 'smart router'}")
    print(f"  Direct AI:  {'skipped' if args.no_direct_ai else 'included'}")

    # Load questions
    print(f"\nLoading questions...")
    try:
        df = pd.read_parquet(questions_path)
    except Exception as e:
        print(f"ERROR: Could not read parquet file: {e}")
        sys.exit(1)

    print(f"Total questions: {len(df)}")
    if len(df) <= sample_size:
        sampled_df = df
        print(f"Using all {len(df)} questions")
    else:
        sampled_df = df.sample(n=sample_size, random_state=42)
        print(f"Sampled {len(sampled_df)} (random_state=42)")

    # Initialise engine
    print(f"\nInitialising engine...")
    try:
        from v24 import build_large_corpus_engine
        db, processor, engine = build_large_corpus_engine(
            db_path=db_path,
            provider=args.provider,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Engine initialisation failed: {e}")
        sys.exit(1)

    # API key check
    print(f"\nChecking API key...")
    try:
        answer, info = engine._generate_answer(
            "Reply with exactly: OK", "", "API test", engine.provider_configs)
        if info.get("provider"):
            print(f"API key valid — using {info['provider'].title()}")
        else:
            print("API key failed — falling back to local extraction.")
    except Exception as e:
        print(f"API key check error: {e}")

    # Run comparison
    print(f"\nRunning comparison on {len(sampled_df)} questions...\n")
    results = []
    query_times = []
    errors = 0

    for idx, (_, row) in enumerate(sampled_df.iterrows(), 1):
        qid = str(row.get("question_id", f"q_{idx}"))
        question = str(row["question"])
        query_start = time.time()

        pct = 100 * idx / len(sampled_df)
        bar_len = 50
        filled = int(bar_len * idx // len(sampled_df))
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\r{bar} {pct:.1f}% ({idx}/{len(sampled_df)})  {qid} ...", end="", flush=True)

        try:
            res = engine.compare(
                question,
                provider=args.provider,
                include_direct_ai=not args.no_direct_ai,
                force_hybrid=args.force_hybrid,
            )
            query_time = time.time() - query_start
            query_times.append(query_time)

            std = res["standard_rag"]
            hyb = res["hybrid_rag"]
            router = hyb.get("router_decision", {})

            gen_info = hyb.get("generation", {})
            status = f"✓ ({query_time:.1f}s)" if gen_info.get("provider") else f"⚠️ local ({query_time:.1f}s)"

            results.append({
                "question_id": qid,
                "question": question[:120],
                "question_type": str(row.get("question_type", "")),
                "std_latency": round(std["latency"], 2),
                "std_tokens": std["tokens_used"],
                "std_relevance": round(std["relevance_score"], 1),
                "std_answer_relevance": round(std.get("answer_relevance", 0), 1),
                "std_chunks": std["chunks_retrieved"],
                "std_confidence": round(std["confidence_score"], 3),
                "hyb_latency": round(hyb["latency"], 2),
                "hyb_tokens": hyb["tokens_used"],
                "hyb_relevance": round(hyb["relevance_score"], 1),
                "hyb_answer_relevance": round(hyb.get("answer_relevance", 0), 1),
                "hyb_chunks": hyb["chunks_retrieved"],
                "hyb_confidence": round(hyb["confidence_score"], 3),
                "hyb_cross_domain": hyb.get("cross_domain", False),
                "hyb_quality_validated": hyb.get("quality_validated", False),
                "router_chosen": router.get("chosen", "unknown"),
                "router_reason": router.get("reason", "")[:80],
                "hybrid_was_skipped": router.get("hybrid_was_skipped", False),
            })
            print(f" {status}")
        except Exception as e:
            query_time = time.time() - query_start
            errors += 1
            print(f" ✗ ({query_time:.1f}s) {str(e)[:80]}")
            continue

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'=' * 70}")

    if not results:
        print("No results to save.")
        sys.exit(1)

    avg_std_rel = sum(r["std_relevance"] for r in results) / len(results)
    avg_hyb_rel = sum(r["hyb_relevance"] for r in results) / len(results)
    avg_std_ar = sum(r["std_answer_relevance"] for r in results) / len(results)
    avg_hyb_ar = sum(r["hyb_answer_relevance"] for r in results) / len(results)
    avg_std_lat = sum(r["std_latency"] for r in results) / len(results)
    avg_hyb_lat = sum(r["hyb_latency"] for r in results) / len(results)
    total_std_tok = sum(r["std_tokens"] for r in results)
    total_hyb_tok = sum(r["hyb_tokens"] for r in results)
    token_diff = total_hyb_tok - total_std_tok
    token_diff_pct = (token_diff / max(1, total_std_tok)) * 100

    hybrid_wins = sum(1 for r in results if r["hyb_answer_relevance"] > r["std_answer_relevance"] + 1)
    standard_wins = sum(1 for r in results if r["std_answer_relevance"] > r["hyb_answer_relevance"] + 1)
    ties = len(results) - hybrid_wins - standard_wins
    cross_domain_count = sum(1 for r in results if r["hyb_cross_domain"])
    hybrid_skipped = sum(1 for r in results if r.get("hybrid_was_skipped", False))

    print(f"\n  Questions run:          {len(results)}")
    print(f"  Errors:                {errors}")
    if hybrid_skipped:
        print(f"  Hybrid skipped by router: {hybrid_skipped}/{len(results)}")
    print(f"  Avg query time:        {sum(query_times)/len(query_times):.1f}s")
    print(f"  Fastest / Slowest:     {min(query_times):.1f}s / {max(query_times):.1f}s")

    print(f"\n  Chunk relevance (term overlap with retrieved chunks):")
    print(f"    Standard avg:  {avg_std_rel:.1f}%")
    print(f"    Hybrid avg:    {avg_hyb_rel:.1f}%")
    print(f"    Difference:    {avg_hyb_rel - avg_std_rel:+.1f} pp")

    print(f"\n  Answer relevance (term overlap with generated answer):")
    print(f"    Standard avg:  {avg_std_ar:.1f}%")
    print(f"    Hybrid avg:    {avg_hyb_ar:.1f}%")
    print(f"    Difference:    {avg_hyb_ar - avg_std_ar:+.1f} pp")

    print(f"\n  Win/Loss/Tie (answer relevance, ±1% threshold):")
    print(f"    Hybrid wins:   {hybrid_wins} ({hybrid_wins/len(results)*100:.0f}%)")
    print(f"    Standard wins: {standard_wins} ({standard_wins/len(results)*100:.0f}%)")
    print(f"    Ties:          {ties} ({ties/len(results)*100:.0f}%)")

    print(f"\n  Latency:")
    print(f"    Standard avg:  {avg_std_lat:.1f}s")
    print(f"    Hybrid avg:    {avg_hyb_lat:.1f}s")
    print(f"    Overhead:      {avg_hyb_lat - avg_std_lat:+.1f}s")

    print(f"\n  Tokens:")
    print(f"    Standard total: {total_std_tok:,}")
    print(f"    Hybrid total:   {total_hyb_tok:,}")
    print(f"    Difference:     {token_diff:+,} ({token_diff_pct:+.1f}%)")

    print(f"\n  Cross-domain triggered: {cross_domain_count}/{len(results)} ({cross_domain_count/len(results)*100:.0f}%)")

    # Save CSV
    csv_path = f"{args.output}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV saved to {csv_path}")

    # Save Markdown
    md_path = f"{args.output}.md"
    lines = [
        "# Standard vs Hybrid RAG — Batch Comparison",
        f"Run on {datetime.now().isoformat()}",
        f"Total questions: {len(results)}",
        f"Database: `{db_path}`",
        f"Questions file: `{questions_path}`",
        f"Provider: {args.provider}",
        f"Mode: {'forced hybrid' if args.force_hybrid else 'smart router'}",
        "",
        "## Summary",
        f"| Metric | Standard | Hybrid | Difference |",
        f"|--------|----------|--------|------------|",
        f"| Chunk relevance | {avg_std_rel:.1f}% | {avg_hyb_rel:.1f}% | {avg_hyb_rel - avg_std_rel:+.1f} pp |",
        f"| Answer relevance | {avg_std_ar:.1f}% | {avg_hyb_ar:.1f}% | {avg_hyb_ar - avg_std_ar:+.1f} pp |",
        f"| Avg latency | {avg_std_lat:.1f}s | {avg_hyb_lat:.1f}s | {avg_hyb_lat - avg_std_lat:+.1f}s |",
        f"| Total tokens | {total_std_tok:,} | {total_hyb_tok:,} | {token_diff:+,} |",
        "",
        f"- **Hybrid wins (answer rel):** {hybrid_wins}/{len(results)} ({hybrid_wins/len(results)*100:.0f}%)",
        f"- **Standard wins:** {standard_wins}/{len(results)} ({standard_wins/len(results)*100:.0f}%)",
        f"- **Ties:** {ties}/{len(results)} ({ties/len(results)*100:.0f}%)",
        f"- **Cross-domain triggered:** {cross_domain_count}/{len(results)}",
    ]
    Path(md_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown report saved to {md_path}")
    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
