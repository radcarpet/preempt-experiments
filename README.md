# preempt
Code submission for NDSS 2026

## Setup
Install libraries in a Python 3.11.4 virtual environment with `requirements.txt`.

## Running Experiments
### Translation and NER
All relevant translation experiment code, results and details can be found under `translation_task`.

Using the [Universal-NER](https://github.com/universal-ner/universal-ner) models requires its own setup. Please follow instructions in the `universal-ner` folder.

### Long-Context Q/A
All relevant long-context Q/A experiment code, results and details can be found under `long_context_task`.

### Multi-turn Financial Q/A
All relevant multi-turn financial Q/A experiment code, results and details can be found under `multi_turn_qa`.

### RAG
Spin up local LLM using:

```
ollama pull mistral:7b-instruct-v0.2-q8_0
```

Run experiments like this, from the root of the repo:

```
python3 scripts/evalqa-rag.py \
    -e qa-cc-zip-date.txt  -nc -o evals  \
    -m litellm/ollama_chat/mistral:7b-instruct-v0.2-q8_0 
    -st aes
```
The `-o` option is the output dir (relative to root of repo) for results.

The last option `-st` option is to specify "sanitization type" (default is "fpe" if omitted).
 - `aes` = AES
 - `rand` = RAND
 - `fpe` = FPE

### Comparisons with PAPILLON
We compare the performance of Preempt with PAPILLON on the translation and long-context Q/A tasks. 

Relevant commands to run PAPILLON on our datasets and corresponding results can be found in the respective task folders.

PAPILLON requires its own libraries for it to run, and can be setup with instructions in the `PAPILLON` folder.
