# Medical Assistant Bot with RAG + Ollama

This project implements a **Retrieval-Augmented Generation (RAG)** system to answer user questions from a structured FAQ dataset. The system integrates **vector search with FAISS** and **large language model (LLM) generation (via Ollama + LangChain)** to provide concise, context-driven answers.

The workflow supports uploading FAQs , preprocessing, creating embeddings, building a retrieval index, and generating answers from retrieved documents only.

---

## Approach

1. **Data Upload & Preprocessing**

   * A CSV containing `question, answer` pairs is uploaded.
   * Each record is assigned a **record\_id** and grouped by **group\_id** (all answers belonging to the same question share a group).
   * The dataset is grouped by exact common questions ensuring balanced question distribution and grouping.

2. **Embedding Model**

   * Experimentation with multiple embedding models, ideally, to choose a model that not only understands the semantics of the language but also emphasizes on the medical teminology and keywords.
   * This project experimented with the following embedding models and found that SapBERT provided the best results:
      - "BAAI/bge-large-en"
      - "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
      - "neuml/pubmedbert-base-embeddings"
      - "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

3. **Index Building (Ingestion)**

   * The training set is embedded using a sentence-transformer embedding model.
   * A **FAISS vector store** is built with embeddings and associated metadata (`record_id, group_id, question, answer`).

4. **Retrieval**

   * Given a user question, the system retrieves top candidate FAQs using FAISS cosine similarity.
   * Configurable parameters:

     * `fetch_k`: number of candidates retrieved from FAISS.
     * `top_k`: max. number of results added to the llm context to generate answer.
     * `similarity_threshold`: minimum similarity cutoff to consider a QnA as valid context.
     * `re_rank`: post fetching step to refine ordering of retrieved documents.

5. **Answer Generation**

   * Retrieved question-answer pairs are passed as **context** to the LLM.
   * The LLM (Ollama, via LangChain) generates an answer using only the provided context.
   * If context is insufficient, the system returns a message indicating that no reliable answer can be generated.

---

##  Assumptions

* **High Accuracy Expected**: Since this project provides a solution to a medical/clinical industry, there is very less margin to hallucination and incorrect answer generation.
* **FAQ style dataset**: Each record consists of a short question and one or more valid answers.
* **Context-bound answers**: The LLM must not use external general knowledge—answers must come from retrieved FAQs only.
* **Embedding quality**: The embedding model sufficiently captures semantic similarity between paraphrased questions as well as appropriately distinguishes between different medical topics.
* **Only Supports English Language**

---

## Model Performance

**Strengths**

* **Semantic retrieval**: Effectively surfaces similar FAQ entries even if user queries are paraphrased.
* **Context control**: Strictly generates answers from retrieved documents, improving reliability.
* **Open Source**: This project uses open source frameworks and data to implement the solution.
* **On-Prem Hosting**: This project does not utilize any premium paid enterprise or cloud app to serve.

**Weaknesses**

* **Quantized**: Since, to ensure that the model is hosted on-prem locally on a macbook 16 gb ram, we needed a 8 bit quantized system and compromise on the floating point precision.
* **Exact copying risk**: LLM sometimes copies retrieved answers instead of synthesizing them.
* **Ambiguity handling**: If retrieved documents contain multiple conflicting answers, the model may not resolve contradictions effectively.
* **Threshold sensitivity**: Performance can degrade if `similarity_threshold` is too strict (few/no docs returned) or too loose (irrelevant docs included).
* **Does not fix misspelt words in the User Query**

---

## Potential Improvements

* **Answer Fusion**: Implement an aggregation step to merge multiple retrieved answers into a more coherent response.
* **Fine-tuning**: Train a domain-specific embedding model for better semantic retrieval.
* **Evaluation Metrics**: Add retrieval metrics (Recall@k, Precision@k, MRR, nDCG) and answer-level metrics (BLEU/ROUGE/F1/METEOR/BERT-Score against reference answers) by hand annotation or natural labeling performance test data.
* **Prompt Optimization**: Refine system prompts to encourage synthesis rather than verbatim copying.
* **Rectify and Enhance User Prompt before Answering**: User Prompts can be misspelt and might not have enough information curated appropriately by the user for expected answer. Similar to chat-gpt, or any other responsive system, enhance and rectify the user prompt before using it for RAG context improvement and answer generation.
* **Increased Floating Point Precision**: With better compute power to harness full potential of the llm model while inferencing, utilize higher/full bit floating point precision.
* **Interactive Feedback Loop**: Allow users to rate answers, improving retrieval weights and answer generation over time to put Human-in-the-loop to validate the model output generation for Completeness, Correctness, Faithfulness.
* **Responsible AI**: Use safety and moderation models like Llama Guard to detect and prevent harmful, unsafe, or inappropriate content in user query and AI-generated responses.

---

## Key Functions

### `upload_csv`

* Uploads a FAQ CSV file.
* Adds `record_id` and `group_id`.

### `ingest_data`

* Loads training data.
* Converts questions into embeddings.
* Builds and stores a FAISS index with metadata.

### `retrieve_similar`

* Retrieves top-k similar FAQ records based on user query.
* Supports configurable `fetch_k`, `top_k`, and `similarity_threshold`.

### `generate_answer`

* Performs RAG pipeline using FAISS retrieval + Ollama generation.
* Builds context prompt and generates an answer strictly from retrieved documents.
* Returns fallback if no sufficient context is found.

# Usage

- source ./setup.sh
- source venv/bin/activate
- uvicorn app.main:app --reload

# Screenshots

## RAG Retrieval
<img width="468" height="244" alt="image" src="https://github.com/user-attachments/assets/258d2340-1e15-4b53-83aa-8059934606bc" />
<img width="468" height="244" alt="image" src="https://github.com/user-attachments/assets/00b9ec2e-05ed-4b07-be3a-6d3f55145fa3" />

```json
{
  "query": "whats glaucoma",
  "results": [
    {
      "record_id": 11415,
      "group_id": 11415,
      "similarity": 0.6990639567375183,
      "question": "What is (are) Glaucoma ?",
      "rerank_score": 3.6911864280700684
    },
    {
      "record_id": 13360,
      "group_id": 13360,
      "similarity": 0.5292325019836426,
      "question": "What is (are) early-onset glaucoma ?",
      "rerank_score": 1.4708786010742188
    },
    {
      "record_id": 5794,
      "group_id": 5794,
      "similarity": 0.5310231447219849,
      "question": "What are the symptoms of Glaucoma ?",
      "rerank_score": 0.9771765470504761
    }
  ]
}
```
## Generate Answer
<img width="468" height="244" alt="image" src="https://github.com/user-attachments/assets/b2e1a6b9-bcaf-4c7c-b7fb-2ee871bea943" />
<img width="468" height="244" alt="image" src="https://github.com/user-attachments/assets/8af56f0f-6b54-4265-9e38-425dac91ee9c" />
<img width="468" height="244" alt="image" src="https://github.com/user-attachments/assets/582b39d1-9929-4d87-8829-b422ec1d7997" />

# Evaluation

1. Retrieval Experimentation.
   - Implementing RAG evaluation metrics will help us to compare the performance of embeddings model and retrieval.
   - It will help us to understand how well does the embedding model emphasize on the cruicial information (eg. disease name), rather than emphasizing on the tokens (eg. "what is", "how are", etc.)
   - Based on the experimentation of multiple embedding models, "bge-large-en", "BioMedBERT", and "PubMedBERT" performed poorly on retrieving correct question from FAISS.
   - However, the embedding model "SapBERT", outperformed the above 3 in those experimentation as shown in the images above and below.

2. RAG Evaluation
   - Data Preparation
      - The FAQ data used to store in the FAISS should also be tagged with information like unique category and saved as metadata.
         o disease name - eg. paget, high-blood-pressure, glaucoma, etc
         o OR
         o further dividing the questions into unique category like disease-interrogatives eg. glaucoma-what, glaucoma-symptoms, glaucoma-diagnosis, glaucoma-prevention, glaucoma-treatments, etc.
      - Now, tag the evaluation test data questions with their respective unique categories.
   - Custom Evaluation Wrapper function
      - will accepts input test query along with expected annotations and the retrieved question records and their annotations.
      - will estimate and return the RAG retrieval evaluation scores.
   - Embedding Model Performance Evaluation
      - Perform estimation of Recall@K, Precision@K, Mean Reciprocal Rank, nDCG based on the retrieved unique categories and test data categories.
      - Now plot all the 3 metrics score across all the embedding models to choose the champion model. In these 4 models, SapBERT outperformed the other 3 as it is not only trained on sentence semantics but also medical terminology emphasis.
   - The hyperparameter tuning to optimize the following retrieval parameters will provide better RAG results:
      - embedding model
      - fetch_k - max no. of records to fetch from vector db for a user query
      - top_k - used ms-marco-MiniLM-L12-v2 to re-rank the fetched records and selected top k.
      - distance calculation - cosine similarity, Approximate Nearest Neighbour (ANN), Hierarchical Navigable Small World (HNSW), etc
<img width="629" height="636" alt="image" src="https://github.com/user-attachments/assets/7f93c969-a15a-4e64-a4c2-84bfeeba095e" />

3. Generation Evaluation
   - Since, there are multiple answers for a question, we are providing all the multiple answers for the question asked by the user to the llm model as context to generate an answer.
   - The generated answer is highly probable to contain longer common sequences from the provided answers in the context window. Hence, generation evaluation metrics like ROUGE-L, METEOR, and BERT-Score would provide better distinguished evaluation scores compared to ROUGE-N gram, BLEU. 
