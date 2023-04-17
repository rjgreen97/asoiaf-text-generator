# asoiaf-text-generator

A pre-trained transformer (`"facebook/opt-125m"` from HuggingFace) fine tuned on the 'Song of Ice and Fire' book series by George R. R. Martin. Once trained, it is able to generate text that features the rich world-building, complex characters, and intricate plotlines that are characteristic of the series. 

The model I used is relatively small in the world of LLMs, so feel free to explore larger models if you have the memory available. In addition, any book series or corpus of text could be used to fine-tune, so don't be afraid to experiment!

# Usage
**1) Install Requirements:**

Built With: `Python 3.10`
```
pip install -r requirements.txt
```
**2) Fine Tune:**

    bin/train.sh --data_filepath --model_name
* `--data_filepath` defaults to a json file containing all the text from the ASOIAF series broken into sentences. Feel free to use your own data instead, which can be prepaired by using `bin/preprocess.sh`. If you are using your own data to fine tune, you will need to alter some filepaths in `utils/preprocessor.py` to match its directory structure.
* `--model_name` defaults to `"facebook/opt-125m"`, but use any model you like.

**3) Generate Text:**

    bin/generate.sh --model_path --model_name
* `--model_path` is the filepath to the saved model checkpoint.
* `--model_name` defaults to `"facebook/opt-125m"`, but make sure to input the appropriate model name if you used something different. 

The terminal will cue a user to submit a promt. The returning output will be the three best results the model generated based on the input prompt.
