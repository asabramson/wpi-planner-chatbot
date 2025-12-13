# CS4342 Final Project - Fine-Tuning a Chatbot for the WPI Planner

See the below guide for information about each file in the repo

* `json_data/`
    * `chatgpt_recommendations.md` - deliverable #2, guidance for model architecture/libraries to use
    * `courses.json` - the data store used by the RAG system where course data is retrieved from
    * `degrees.json` - the data store used by the RAG system where degree program data is retrieved from
    * `export.py` - the scraping script used to pull course data from WPI's server
    * `fine_tuning_transformed.json` - the fine-tuning prompts used with `unsloth` (`get_info()` function replaced)
    * `fine_tuning_unformatted.json` - the fine-tuning prompts before being formatted (`get_info() placeholder still present)
    * `prod-data-raw.json` - an example of what the raw scraped course data looks like (used by `export.py`)
    * `transform_prompts.py` - the script used to transform `fine_tuning_unformatted.json` into `fine_tuning_transformed`
    * `wpi-info.json` - contains information about WPI, populated `degrees.json`
* `chat_history.txt` - full history of each of our 18 conversations, which contain a query from the student, a reponse from the model, and a confidence score
    * Higher confidence scores (closer to 0) correspond to the model having more confidence in its response, lower confidence scores (more negative) correspond to the model having less confidence
* `download_model.py` - used for testing `huggingface_hub`, which is used when pulling the pre-trained model
* `fine_tuning.py` - takes a pre-trained model (Llama-3.3-70B-Instruct-bnb-4bit), fine-tunes with `fine_tuning_transformed.json` with the `unsloth` library, saves the model as a `.gguf` file
* `input_parser.py` - helper script which parses the user input for mentions of a course or degree program, their respective data is then pulled from `courses.json` or `degrees.json` and passed to the model during inference
* `loss_data.txt` - loss, grad_norm, learning rate, and epoch information from the fine-tuning process
* `model_inference` - loads the model from a `.gguf` file, uses `llama-cpp-python` to run the model, before user input is passed into the model `input_parser.py` retrieves relevant coruse/degree information, a 3D PCA plot is generated (`pca_graph.png`)
* `pca_graph.png` - compares variations in the input to the model's confidence in its response
* `test_model.py` - used for testing `llama-cpp-python`
* `test_unsloth.py` - used for testing `unsloth`
* `training_loss_documentation.png` - displays the token-level cross entropy loss over the 60 steps of fine-tuning