# muVERA: A Python Implementation of Multi-Vector Retrieval

This project is a Python implementation of a multi-vector document retrieval and re-ranking system, inspired by the concepts in the muVERA research paper. It uses a two-stage process to efficiently search a corpus of documents for the most relevant results for a given query.

## Setup and Usage

### Dependencies

You will need the following Python libraries. You can install them using pip:

```bash
pip install -r requirements.txt
```


### Running the Demo

To run the example, simply execute the `main.py` script:

```bash
python main.py
```

This will index the sample corpus, run the query "fast animals", and print the top 3 most relevant documents with their scores.

