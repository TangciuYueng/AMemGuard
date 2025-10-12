## :rocket: Validation Experiments

Follow these steps to run the validation experiments.
We use knowledge graph to visualize our findings that benign and malicious memory have different characteristics.

### 1\. Prepare the Data

First, download the dataset required for the experiments from the link below.

**Data Source**: [AgentAuditor-ASSEBench](https://github.com/Astarojth/AgentAuditor-ASSEBench/)

### 2\. Set Up the Environment

We strongly recommend using `conda` to create an isolated virtual environment to avoid package conflicts.

```bash
conda create -n validation_env python=3.10

conda activate validation_env
```

### 3\. Install Dependencies

Once the environment is activated, install all the required packages using `pip`.

```bash
pip install -r requirements.txt
```

### 4\. Run the Validation Script

```bash
bash run_validation.sh
```
