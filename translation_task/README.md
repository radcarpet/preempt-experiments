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

---
### Comparisons with PAPILLON

PAPILLON's responses for the translation task and corresponding BLEU scores can be found under `preempt/PAPILLON/results` as `papillon_translation_task_{Age/Name/Money}_{de/fr}.csv` and `papillon_translation_task_{Age/Name/Money}_{de/fr}.json` respectively. 

- Optimized prompts can be found under `preempt/PAPILLON/papillon/optimized_prompts/`.

PAPILLON can be run on the translation task as follows:
1. Export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>" and go to `preempt/PAPILLON/`
2. Host Llama-3.1 8B Instruct through SGLang:
  - ```python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --port <PORT_NUMBER>```
3. Optimize pipeline for translation:
  - ```
    cd papillon
    
    python3 run_dspy_optimization_llama.py --port <PORT_NUMBER> --prompt_output "papillon_translation_task_optimized_prompt_{de/fr}_{age/name/money}.json" --data_file "papillon_finetuning_translation_task_processed_{de/fr}_{age/name/money}.csv"
    ```
4. Evaluate pipeline:
- ```
  python3 evaluate_papillon.py --port <PORT_NUMBER> --model_name <MODEL_NAME> (e.g. meta-llama/Llama-3.1-8B-Instruct) --data_file "papillon_translation_task_{Age/Name/Money}_{de/fr}.csv"
  ```
