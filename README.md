# Masai_AIML_FinalProject
Overview

This repository contains three Python-based modules covering different stages of data and AI work:

Books Web Scraping & Database Project – collects book data, cleans it, stores it in CSV/SQLite, and performs SQL and Pandas analysis.

Titanic Data Analysis & Machine Learning Project – performs EDA and then builds and evaluates machine learning models.

Zepto GenAI RAG Service – provides a policy question-answering API using embeddings, vector search, LangGraph, Pydantic, and FastAPI.

Together, the modules demonstrate a basic end-to-end progression from data collection and analysis to machine learning and GenAI application development.

Modules

1. Books Web Scraping & Database

This module demonstrates a simple data-engineering workflow:

Web Scraping → Data Cleaning → CSV → SQLite Database → SQL/Pandas Analysis

It scrapes book information from Books to Scrape, converts and cleans the data, stores it in a normalized SQLite database, and performs basic queries and analysis. 

2. Titanic EDA & Machine Learning

This module focuses on understanding a dataset and applying machine learning:

Data Loading → EDA → Data Cleaning → Visualization → Feature Preparation → Model Training → Model Evaluation

The EDA script prepares and explores the Titanic dataset, while the modeling script uses the prepared data to train and evaluate machine learning models. 

3. Zepto GenAI RAG Service

This module demonstrates a basic Retrieval-Augmented Generation application for answering Zepto policy questions.

It uses policy documents, local embeddings, ChromaDB for vector retrieval, LangGraph for workflow orchestration, Pydantic for structured responses, and FastAPI for the API layer. 

The service supports policy-question retrieval and a mock/offline mode for deterministic testing. fileciteturn3file2L118-L138

High-Level Project Flow

Module 1
Web Scraping
     ↓
Data Cleaning
     ↓
CSV / SQLite
     ↓
SQL & Pandas Analysis

Module 2
Titanic Dataset
     ↓
EDA & Visualization
     ↓
Data Preparation
     ↓
Machine Learning
     ↓
Model Evaluation

Module 3
Zepto Policies
     ↓
Embeddings
     ↓
Vector Database
     ↓
Retrieval
     ↓
LangGraph Workflow
     ↓
FastAPI
     ↓
Policy Answer

Technologies Used

Python

Pandas

NumPy

Requests

BeautifulSoup

SQLite / SQL

Matplotlib

Seaborn

Scikit-learn

Imbalanced-learn

Joblib

Sentence Transformers

ChromaDB

LangGraph

Pydantic

FastAPI

Getting Started

Install the dependencies required for the individual modules according to their respective requirements.

Run each module independently using its Python entry point.

The exact setup and execution steps can be maintained in the individual module documentation.

Summary

This project provides a high-level demonstration of three practical areas:

Data Engineering: scraping, cleaning, database storage, and querying.

Data Science / Machine Learning: EDA, preprocessing, model training, and evaluation.

Generative AI: document retrieval, embeddings, RAG workflow, and API development.
