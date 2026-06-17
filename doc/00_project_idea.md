# Project Idea

## Problem

RideFlow NN already has forecasting, routing, and pricing logic, but a reviewer or user needs a quick way to ask natural-language questions about how the system works. Reading separate code, README sections, and report notes is slower than querying an indexed knowledge base.

## Goal

Build a small educational RAG system over RideFlow NN project materials. The system should answer questions about demand forecasting, route-based pricing, data sources, dashboard behavior, limitations, and generated artifacts.

## Audience

The audience is a course reviewer, student, or project maintainer who wants to inspect RideFlow behavior without reading the whole codebase first.

## Data

The corpus is `data/raw/datasets.json`. It contains 1212 text records: 12 overview records assembled from local RideFlow project documentation and 1200 text records generated from the project's Moscow hourly demand CSV. This satisfies the excellent-grade scale criterion of 1000+ source records.

## Success Criteria

- The project has an end-to-end RAG pipeline: ingest, chunking, TF-IDF index, retrieval, answer generation, and Streamlit UI.
- Relevant RideFlow questions return an answer with sources.
- Unrelated questions return a refusal without hallucinated facts.
- Tests for chunking, retrieval, and answer policy pass.
