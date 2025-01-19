# Translation

Translation experiments can be run using `translation_jobs_uniner_de.sh`.
- The NER model path can be changed to Llama-3 8B Instruct, Gemma-2 9B Instruct, or UniNER.
- The task by default is English->German. This can be changed to English->French by setting `TASK=fr-en`.
- The translation model is set by default as OPUS-MT (the path must be updated within the script [here](https://github.com/radcarpet/preempt/blob/0abe6e077d214f44682411bcd47194de4f409e79/translation_task/translation_task.py#L103)).
- Online API models like GPT-4 and Gemini-v1.5 can be used by setting `TRANS_MODEL=gpt-4` and `TRANS_MODEL=gemini` respectively. The API key must be provided in the `API_KEY` field.
- Cached data in `cache` is used to run the experiment, in the format `en_{de/fr}_data_{Name/Age/Money}.json`

# NER 

NER experiments can be run using `ner_extract_jobs.sh`.
- The NER model path can be set with the `MODEL` field, including closed-source models (GPT-4 Turbo, Gemini-1.5, Claude 3.5 Sonnet).
- This uses cached data found under `results/ner_results`.
