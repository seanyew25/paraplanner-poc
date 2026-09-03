"""Extract archetype guides into SQLite and the archetype_guides Chroma collection."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import re
import sqlite3
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class BenchmarkRules(BaseModel):
    emergency_fund_months_min: Optional[int] = Field(default=None)
    emergency_fund_months_max: Optional[int] = Field(default=None)
    emergency_fund_irregular_months: Optional[int] = Field(default=None)
    death_tpd_income_multiplier: Optional[float] = Field(default=None)
    ci_income_multiplier: Optional[float] = Field(default=None)
    max_insurance_spend_percent: Optional[float] = Field(default=None)
    min_investment_percent: Optional[float] = Field(default=None)


class NationalSchemeGuidance(BaseModel):
    scheme_name: str
    applicable_status: str
    guidance_notes: str


class ArchetypeSemanticChunk(BaseModel):
    chunk_id: str
    category: str
    guidance_text: str
    topic: Optional[str] = None


class ParsedArchetypeGuide(BaseModel):
    archetype_id: str
    archetype_name: str
    target_age_min: int
    target_age_max: int
    family_and_dependent_context: str
    benchmarks: BenchmarkRules
    relevant_national_schemes: list[NationalSchemeGuidance] = Field(default_factory=list)
    semantic_chunks: list[ArchetypeSemanticChunk] = Field(default_factory=list)


EXTRACTION_PROMPT = """
Analyze this Singapore Basic Financial Planning Guide and return the supplied JSON schema.
The source filename is {source_filename}. Use the stable archetype ID {archetype_id}.

Extraction rules:
1. Extract the exact archetype name and inclusive target age boundaries.
2. Extract numerical rules of thumb into benchmarks. Use null when a value is not stated;
   do not infer or invent a threshold from a case study.
3. Distinguish general benchmarks from case-study-specific values. Case studies belong in
   contextual chunks and must not replace general benchmark values.
4. Capture national scheme guidance such as MediShield Life, CareShield Life, DPS, CPF,
   CPF LIFE, and MRSS when present.
5. Create self-contained semantic chunks grouped by Emergency, Protection, Investments,
   Legacy, Housing, or another clear life-planning category.
6. Preserve qualifying conditions, age brackets, exceptions, warnings, and named schemes.
7. Keep related advice together. Do not split a case study, rule with its age bracket,
   or a recommendation from its conditions.
""".strip()


def archetype_id_for_path(pdf_path: Path) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", pdf_path.stem).strip("_").upper()
    normalized = normalized.replace("ENGLISH_", "")
    return normalized[:80]


def create_archetype_tables(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS archetypes (
            archetype_id TEXT PRIMARY KEY,
            archetype_name TEXT NOT NULL,
            target_age_min INTEGER NOT NULL,
            target_age_max INTEGER NOT NULL,
            family_and_dependent_context TEXT NOT NULL,
            source_filename TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS archetype_benchmarks (
            archetype_id TEXT PRIMARY KEY,
            emergency_fund_months_min INTEGER,
            emergency_fund_months_max INTEGER,
            emergency_fund_irregular_months INTEGER,
            death_tpd_income_multiplier REAL,
            ci_income_multiplier REAL,
            max_insurance_spend_percent REAL,
            min_investment_percent REAL,
            FOREIGN KEY (archetype_id) REFERENCES archetypes(archetype_id)
                ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS archetype_schemes (
            scheme_id INTEGER PRIMARY KEY AUTOINCREMENT,
            archetype_id TEXT NOT NULL,
            scheme_name TEXT NOT NULL,
            applicable_status TEXT NOT NULL,
            guidance_notes TEXT NOT NULL,
            FOREIGN KEY (archetype_id) REFERENCES archetypes(archetype_id)
                ON DELETE CASCADE,
            UNIQUE (archetype_id, scheme_name)
        );

        CREATE INDEX IF NOT EXISTS idx_archetypes_age
            ON archetypes(target_age_min, target_age_max);
        """
    )
    connection.commit()


def save_to_sqlite(
    guide: ParsedArchetypeGuide,
    source_filename: str,
    connection: sqlite3.Connection,
) -> None:
    benchmark = guide.benchmarks
    connection.execute("DELETE FROM archetype_schemes WHERE archetype_id = ?", (guide.archetype_id,))
    connection.execute("DELETE FROM archetype_benchmarks WHERE archetype_id = ?", (guide.archetype_id,))
    connection.execute("DELETE FROM archetypes WHERE archetype_id = ?", (guide.archetype_id,))
    connection.execute(
        """
        INSERT INTO archetypes (
            archetype_id, archetype_name, target_age_min, target_age_max,
            family_and_dependent_context, source_filename
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            guide.archetype_id,
            guide.archetype_name,
            guide.target_age_min,
            guide.target_age_max,
            guide.family_and_dependent_context,
            source_filename,
        ),
    )
    connection.execute(
        """
        INSERT INTO archetype_benchmarks (
            archetype_id, emergency_fund_months_min, emergency_fund_months_max,
            emergency_fund_irregular_months, death_tpd_income_multiplier,
            ci_income_multiplier, max_insurance_spend_percent, min_investment_percent
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            guide.archetype_id,
            benchmark.emergency_fund_months_min,
            benchmark.emergency_fund_months_max,
            benchmark.emergency_fund_irregular_months,
            benchmark.death_tpd_income_multiplier,
            benchmark.ci_income_multiplier,
            benchmark.max_insurance_spend_percent,
            benchmark.min_investment_percent,
        ),
    )
    connection.executemany(
        """
        INSERT INTO archetype_schemes (
            archetype_id, scheme_name, applicable_status, guidance_notes
        ) VALUES (?, ?, ?, ?)
        """,
        [
            (
                guide.archetype_id,
                scheme.scheme_name,
                scheme.applicable_status,
                scheme.guidance_notes,
            )
            for scheme in guide.relevant_national_schemes
        ],
    )
    connection.commit()


def save_to_chroma(guide: ParsedArchetypeGuide, source_filename: str, collection) -> None:
    if not guide.semantic_chunks:
        return

    collection.upsert(
        ids=[f"{guide.archetype_id}_{chunk.chunk_id}" for chunk in guide.semantic_chunks],
        documents=[chunk.guidance_text for chunk in guide.semantic_chunks],
        metadatas=[
            {
                "archetype_id": guide.archetype_id,
                "archetype_name": guide.archetype_name,
                "category": chunk.category,
                "chunk_id": chunk.chunk_id,
                "source_filename": source_filename,
            }
            for chunk in guide.semantic_chunks
        ],
    )


def extract_guide(pdf_path: Path, api_key: str) -> ParsedArchetypeGuide:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    uploaded_file = client.files.upload(file=str(pdf_path))
    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=[
                uploaded_file,
                EXTRACTION_PROMPT.format(
                    source_filename=pdf_path.name,
                    archetype_id=archetype_id_for_path(pdf_path),
                ),
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ParsedArchetypeGuide,
                temperature=0.0,
            ),
        )
        if response.parsed is None:
            raise ValueError("Gemini returned no parsed structured response")
        return ParsedArchetypeGuide.model_validate(response.parsed)
    finally:
        client.files.delete(name=uploaded_file.name)


def ingest_directory(
    guides_dir: Path,
    database_path: Path,
    chroma_path: Path,
    max_workers: int,
) -> tuple[int, list[tuple[Path, str]]]:
    import chromadb

    pdf_paths = sorted(guides_dir.rglob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found under {guides_dir}")

    api_key = os.getenv("API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing API_KEY or GEMINI_API_KEY environment variable")

    chroma_client = chromadb.PersistentClient(path=str(chroma_path))
    chroma_client.get_or_create_collection(name="policy_clauses")
    archetype_collection = chroma_client.get_or_create_collection(name="archetype_guides")

    processed = 0
    failures: list[tuple[Path, str]] = []
    with sqlite3.connect(database_path) as connection:
        create_archetype_tables(connection)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(extract_guide, pdf_path, api_key): pdf_path
                for pdf_path in pdf_paths
            }
            for future in as_completed(futures):
                pdf_path = futures[future]
                try:
                    guide = future.result()
                    save_to_sqlite(guide, pdf_path.name, connection)
                    save_to_chroma(guide, pdf_path.name, archetype_collection)
                    processed += 1
                    print(
                        f"Processed {pdf_path.name} -> {guide.archetype_id} "
                        f"({len(guide.semantic_chunks)} chunks)"
                    )
                except Exception as error:
                    failures.append((pdf_path, str(error)))
                    print(f"Failed {pdf_path.name}: {error}")

    return processed, failures


def main() -> int:
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--guides-dir",
        type=Path,
        default=Path(__file__).parents[1] / "archetype_guides",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path(__file__).with_name("policies.db"),
    )
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=Path(__file__).with_name("chroma_db"),
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")

    load_dotenv(Path(__file__).parents[1] / ".env")
    processed, failures = ingest_directory(
        args.guides_dir,
        args.database,
        args.chroma_dir,
        args.workers,
    )
    print(f"Finished: {processed} processed, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())