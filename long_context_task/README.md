# Long-Context Q/A

Long-context Q/A experiments can be run using `long_QA_jobs.sh`.
- The NER model path can be changed to Llama-3 8B Instruct, Gemma-2 9B Instruct, or UniNER.
- The QA model can be set with `QA_MODEL={gpt-4/gemini/path_to_llama3}`.
- A model is required for semantic textual similarity, which can be obtained from [here](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- Results are found in the `results` folder, with the format: `{ner_model}_ner_extract_{qa_model}_Name_0_50_{samples}`.

---
### Comparisons with PAPILLON

PAPILLON's responses for the long-context Q/A task and corresponding Semantic Textual Similarity scores can be found under `preempt/PAPILLON/results` as `papillon_long_context_task.csv` and `papillon_long_context_task.json` respectively.

- Optimized prompts can be found under `preempt/PAPILLON/papillon/optimized_prompts/`.

PAPILLON can be run on the long-context Q/A task as follows:
1. Export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>" and go to `preempt/PAPILLON/`
2. Host Llama-3.1 8B Instruct through SGLang:
  - ```python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --port <PORT_NUMBER>```
3. Optimize pipeline for long-context Q/A:
  - ```
    cd papillon
    
    python3 run_dspy_optimization_llama.py --port <PORT_NUMBER> --prompt_output "papillon_long_context_task_optimized_prompt.json" --data_file "papillon_finetuning_long_context_task_processed.csv"
    ```
4. Evaluate pipeline:
- ```
  python3 evaluate_papillon.py --port <PORT_NUMBER> --model_name <MODEL_NAME> (e.g. meta-llama/Llama-3.1-8B-Instruct) --data_file "papillon_long_context_task.csv"
  ```
