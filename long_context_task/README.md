# Long-Context Q/A

Long-context Q/A experiments can be run using `long_QA_jobs.sh`.
- The NER model path can be changed to Llama-3 8B Instruct, Gemma-2 9B Instruct, or UniNER.
- The QA model can be set with `QA_MODEL={gpt-4/gemini/path_to_llama3}`.
- A model is required for semantic textual similarity, which can be obtained from [here](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- Results are found in the `results` folder, with the format: `{ner_model}_ner_extract_{qa_model}_Name_0_50_{samples}`.
