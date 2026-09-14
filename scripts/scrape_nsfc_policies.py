#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scrape NSFC policy guide documents and prepare RAG-ready data."""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.scrapers.data_processor import process_nsfc_data
from src.scrapers.nsfc_scraper import NSFCScraper


async def run_scrape(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    async with NSFCScraper(
        max_pages=args.pages,
        delay=args.delay,
        use_selenium=args.selenium,
        list_url=args.list_url,
    ) as scraper:
        documents = await scraper.scrape_all_pages()

    if not documents:
        print("No documents were scraped.")
        return 1

    raw_dir = PROJECT_ROOT / args.raw_dir
    processed_dir = PROJECT_ROOT / args.processed_dir

    async with NSFCScraper() as saver:
        await saver.save_to_files(documents, str(raw_dir))

    processed = await process_nsfc_data(documents, str(processed_dir))

    stats = processed["stats"]
    print("Scrape and processing complete.")
    print(f"Raw documents: {len(documents)}")
    print(f"Valid documents: {stats['total_documents']}")
    print(f"Chunks: {stats['total_chunks']}")
    print(f"Avg chunks/doc: {stats['avg_chunks_per_doc']:.2f}")
    print(f"Raw output: {raw_dir}")
    print(f"Processed output: {processed_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape NSFC policy guide documents.")
    parser.add_argument("--pages", type=int, default=6, help="Maximum list pages to scrape.")
    parser.add_argument("--delay", type=float, default=1.0, help="Request delay in seconds.")
    parser.add_argument("--selenium", action="store_true", help="Use Selenium instead of aiohttp.")
    parser.add_argument(
        "--list-url",
        default=None,
        help="NSFC list URL. Defaults to the current 2025 guide list.",
    )
    parser.add_argument(
        "--raw-dir",
        default="data/nsfc_policy_docs_2025",
        help="Directory for scraped markdown files.",
    )
    parser.add_argument(
        "--processed-dir",
        default="data/nsfc_policy_processed_2025",
        help="Directory for processed RAG-ready JSON files.",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    return parser.parse_args()


def main() -> None:
    raise SystemExit(asyncio.run(run_scrape(parse_args())))


if __name__ == "__main__":
    main()
