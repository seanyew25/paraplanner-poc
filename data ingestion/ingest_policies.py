"""Extract policy metadata and context-preserving chunks from PDFs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import re
import sqlite3
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field


class SQLMetadata(BaseModel):
    product_name: str = Field(description="Name of the insurance product")
    insurer: str = Field(description="Name of the insurer")
    category: str = Field(description="Term Life, Whole Life, or Critical Illness Rider")
    participating: bool = Field(description="Whether the policy participates in bonuses")
    has_cash_value: bool = Field(description="Whether the policy develops surrender value")
    min_entry_age: Optional[int] = Field(default=None, description="Minimum age next birthday")
    max_entry_age: Optional[int] = Field(default=None, description="Maximum age next birthday")
    policy_terms: List[str] = Field(default_factory=list, description="Available policy terms")
    covered_events: List[str] = Field(default_factory=list, description="Covered insured events")
    max_tpd_benefit: Optional[float] = Field(default=None, description="Maximum TPD benefit")


class VectorChunk(BaseModel):
    chunk_id: str = Field(description="Unique identifier within this document")
    section: str = Field(description="Benefits, Exclusions, Riders, Termination, or Premiums")
    topic: str = Field(description="Specific clause topic")
    text_content: str = Field(description="Verbatim or near-verbatim policy wording")


class ProcessedPolicyDocument(BaseModel):
    policy_id: str = Field(description="Short stable identifier for this document")
    sql_record: SQLMetadata
    vector_chunks: List[VectorChunk] = Field(default_factory=list)


EXTRACTION_PROMPT = """
Extract this insurance Product Summary into the supplied JSON schema.
The stable document ID is {policy_id}. Use it for policy_id.

CRITICAL CHUNKING RULES:
1. Do not split exclusion lists or benefit schedules across chunks.
2. Keep waiting periods and pre-existing-condition terms fully intact.
3. If the document covers a base plan and attached riders, create distinct rider chunks.
4. Preserve verbatim definitions for TPD, Terminal Illness, and Critical Illness payouts.
5. Include all relevant conditions, exceptions, lists, caps, and termination rules in the
   chunk where they belong. Do not summarize away legal conditions.
6. Create useful chunks for definitions, benefits, exclusions, waiting periods, riders,
   termination, and premiums. A chunk must stand alone when retrieved from the database.
""".strip()


def policy_id_for_path(pdf_path: Path) -> str:
    """Return a stable, SQLite-safe ID derived from the source filename."""
    normalized = re.sub(r"[^A-Za-z0-9]+", "-", pdf_path.stem).strip("-").upper()
    return f"POL-{normalized[:75]}"


def create_database(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS policies (
            policy_id TEXT PRIMARY KEY,
            product_name TEXT NOT NULL,
            insurer TEXT NOT NULL,
            category TEXT NOT NULL,
            participating INTEGER NOT NULL,
            has_cash_value INTEGER NOT NULL,
            min_entry_age INTEGER,
            max_entry_age INTEGER,
            policy_terms TEXT NOT NULL,
            covered_events TEXT NOT NULL,
            max_tpd_benefit REAL
        )
        """
    )
    connection.commit()


def save_to_stores(
    parsed_doc: ProcessedPolicyDocument,
    connection: sqlite3.Connection,
    collection,
) -> None:
    """Replace one policy row and all of its vector chunks atomically per store."""
    import json

    meta = parsed_doc.sql_record
    connection.execute(
        """
        INSERT OR REPLACE INTO policies (
            policy_id, product_name, insurer, category, participating,
            has_cash_value, min_entry_age, max_entry_age, policy_terms,
            covered_events, max_tpd_benefit
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            parsed_doc.policy_id,
            meta.product_name,
            meta.insurer,
            meta.category,
            int(meta.participating),
            int(meta.has_cash_value),
            meta.min_entry_age,
            meta.max_entry_age,
            json.dumps(meta.policy_terms),
            json.dumps(meta.covered_events),
            meta.max_tpd_benefit,
        ),
    )
    connection.commit()

    if not parsed_doc.vector_chunks:
        return

    collection.upsert(
        documents=[chunk.text_content for chunk in parsed_doc.vector_chunks],
        metadatas=[
            {
                "policy_id": parsed_doc.policy_id,
                "product_name": meta.product_name,
                "section": chunk.section,
                "topic": chunk.topic,
                "category": meta.category,
            }
            for chunk in parsed_doc.vector_chunks
        ],
        ids=[f"{parsed_doc.policy_id}_{chunk.chunk_id}" for chunk in parsed_doc.vector_chunks],
    )


def extract_policy(pdf_path: Path, client, policy_id: str) -> ProcessedPolicyDocument:
    """Upload one PDF, parse it with Gemini, and always delete the remote file."""
    uploaded_file = client.files.upload(file=str(pdf_path))
    try:
        from google.genai import types

        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=[uploaded_file, EXTRACTION_PROMPT.format(policy_id=policy_id)],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ProcessedPolicyDocument,
                temperature=0.0,
            ),
        )
        parsed = response.parsed
        if parsed is None:
            raise ValueError("Gemini returned no parsed structured response")
        parsed.policy_id = policy_id
        return parsed
    finally:
        client.files.delete(name=uploaded_file.name)


def ingest_directory(
    policies_dir: Path,
    database_path: Path,
    chroma_path: Path,
    max_workers: int = 4,
) -> tuple[int, list[tuple[Path, str]]]:
    from google import genai
    import chromadb

    pdf_paths = sorted(policies_dir.rglob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found under {policies_dir}")

    api_key = os.getenv("API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing API_KEY or GEMINI_API_KEY environment variable")

    chroma_client = chromadb.PersistentClient(path=str(chroma_path))
    collection = chroma_client.get_or_create_collection(name="policy_clauses")

    processed = 0
    failures: list[tuple[Path, str]] = []
    with sqlite3.connect(database_path) as connection:
        create_database(connection)
        def extract_from_path(pdf_path: Path) -> ProcessedPolicyDocument:
            client = genai.Client(api_key=api_key)
            return extract_policy(pdf_path, client, policy_id_for_path(pdf_path))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(extract_from_path, pdf_path): pdf_path for pdf_path in pdf_paths}
            for future in as_completed(futures):
                pdf_path = futures[future]
                try:
                    parsed_doc = future.result()
                    save_to_stores(parsed_doc, connection, collection)
                    processed += 1
                    print(
                        f"Processed {pdf_path.name} -> {parsed_doc.policy_id} "
                        f"({len(parsed_doc.vector_chunks)} chunks)"
                    )
                except Exception as error:
                    failures.append((pdf_path, str(error)))
                    print(f"Failed {pdf_path.name}: {error}")

    return processed, failures


def main() -> int:
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policies-dir", type=Path, default=Path(__file__).parents[1] / "Policies")
    parser.add_argument("--database", type=Path, default=Path(__file__).with_name("policies.db"))
    parser.add_argument("--chroma-dir", type=Path, default=Path(__file__).with_name("chroma_db"))
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Maximum number of PDFs processed concurrently (default: 4)",
    )
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")

    load_dotenv(Path(__file__).parents[1] / ".env")
    processed, failures = ingest_directory(args.policies_dir, args.database, args.chroma_dir, args.workers)
    print(f"Finished: {processed} processed, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())