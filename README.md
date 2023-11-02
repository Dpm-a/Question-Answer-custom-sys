# Dependencies

It is recommended installing a fresh Conda environment and run python 3.8+.

```bash
conda create --name <my_env> python==3.8
conda activate <my_env>
```

Once activated, we need to install dependencies:

```bash
git clone https://github.com/Dpm-a/stip_openai
cd src/
source requirements.sh
```

# Usage

In `Notebook` there will be a `.ipynb` file use for different trials on input data.

The only file to run is instead:

```bash
 cd src/
 python gpt_qa.py -q "your_query"
```

## Optional Parameters

The script supports several optional parameters to customize the behavior. These parameters can be added when running the script from the command line. Below is the list of optional parameters:

- `-gpt4` or `--gpt4`: Choose between GPT-3.5 and GPT-4. If included, the script will use GPT-4; otherwise, it will default to GPT-3.5.

- `-ft` or `--fine_tuning`: Use Fine Tune Model instead of QA one. Include this flag to enable fine-tuning mode and eventually also ignore GPT4.

- `-p` or `--prints`: Print ranked messages. If set to `True`, the script will print ranked messages.

### Usage Example

To run the script with optional parameters, use the following format:

```bash
cd src/
python gpt_qa.py -q "your_query" -gpt4 -ft -p
```
