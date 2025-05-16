Script for generation of results for ARC prompting. 

src/openrouter_api_arc.py

- Sends the ARC eval set to the openrouter API - requires Openrouter API key. 

src/batch_api.py

- Sends the ARC eval set to the openapi batch processing API - requires OpenAI API key. 

src/sft_model.py

- Supervised fine tuning approach of opensource models on ARC data - requires huggingface token. Also requires significant VRAM. 

src/evaluation.py

- Evaluate returned text from the API model against out metrics


Scripts Folder

- Scripts to run SFT/Baseline models on the HPC. 


Data

All JSONL files contain the model that the query was ran on, and the associated task_id for that prompt.


data/filtered_sft/arc_cots_full_eval.jsonl

- Contains correct and incorrect replies to the ARC eval dataset from various Openrouter model using COT. 184 Correct COT sequences and 1224 incorrect ones. 

data/generated_sft/prompt_engineering

- Contains raw data for the experiements ran in this paper. Each file is prepended with the approach - i.e. {approach}_{dataset}_{model.jsonl}.
