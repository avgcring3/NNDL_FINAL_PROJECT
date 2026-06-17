# Project Idea

## Problem

RideFlow NN already has forecasting, routing, and pricing logic, but a reviewer or user needs a quick way to ask natural-language questions about how the system works. Reading separate code, README sections, and report notes is slower than querying an indexed knowledge base.

## Goal

Build a small educational RAG system over RideFlow NN project materials. The system should answer questions about demand forecasting, route-based pricing, data sources, dashboard behavior, limitations, and generated artifacts.

## Audience

The audience is a course reviewer, student, or project maintainer who wants to inspect RideFlow behavior without reading the whole codebase first.

## Data

The MVP corpus is `data/raw/datasets.json`. It contains 12 text records assembled from the local RideFlow project description and implementation notes. This satisfies the homework MVP minimum of 10 text records.

## Success Criteria

- The project has an end-to-end RAG pipeline: ingest, chunking, TF-IDF index, retrieval, answer generation, and Streamlit UI.
- Relevant RideFlow questions return an answer with sources.
- Unrelated questions return a refusal without hallucinated facts.
- Tests for chunking, retrieval, and answer policy pass.
