# Relapse Death Screening from Multilingual Clinical Notes

A two-agent system built with Google's Agent Development Kit (ADK). It reads a short clinical note written in any language and answers one question:

**Does the text say that the patient died of a relapse? Yes or no.**

> **Disclaimer**
>
> - **Course material:** This project was built for the Google and Kaggle 5-Day Agents course. Parts of the code and structure are reused or adapted from that course's notebooks. 
> - **AI assistance:** The Markdown explanations in the notebook and this README were written and polished with AI assistance and reviewed by me.
> - **Not for clinical use:** This is a learning project and a screening helper, not a clinical decision tool. All example notes are invented.

## How it works

| Agent | Job |
|---|---|
| TranslatorAgent | Translates the note into English, keeping negations and medical terms |
| RelapseDeathClassifierAgent | Calls a tool that finds death and relapse sentences, then answers `ANSWER: YES` or `ANSWER: NO` |

The rule: answer `YES` only if the text explicitly says the patient died and attributes the death to a relapse or recurrence. Everything else is `NO`, including "cause of death not stated".


## What the notebook covers

| Section | Topic |
|---|---|
| 1 | Setup |
| 2 | Invented test notes in six languages with expected answers |
| 3 | Translator agent |
| 4 | Classifier agent with a custom tool |
| 5 | Connecting the agents (`SequentialAgent`) |
| 6 | Sessions, persistent storage, state, context compaction |
| 7 | Long-term memory |
| 8 | Human-in-the-loop approval |
| 9 | Observability: logs, plugins, metrics |
| 10 | Packaging as an ADK project (`adk web`) |
| 11 | Evaluation |
| 12 | Deployment to Vertex AI Agent Engine (not run) |

The notebook is saved with its outputs, so results can be read on GitHub without running anything.

## Results

On the seven invented test notes the pipeline answered all seven correctly (7/7, 100%).

The test set is small and invented, so this does not show real-world accuracy.

## Repository structure

```
.
|-- multilingual-relapse-screening-agents.ipynb   main notebook (with saved outputs)
|-- relapse_agent/                                ADK project folder
|   |-- __init__.py
|   |-- agent.py                                  translator, tool, classifier, root_agent
|   |-- test_config.json                          adk eval configuration
|   `-- integration.evalset.json                  adk eval test cases
|-- relapse_agent_deploy/                         prepared deployment folder (not deployed)
|-- requirements.txt
|-- .env.example
|-- .gitignore
`-- README.md
```

## Setup

You need Python 3.10 or newer (developed with Python 3.13) and a Gemini API key from [Google AI Studio](https://aistudio.google.com/app/api-keys). You also need to install required libraries. In .env file:

```
GOOGLE_API_KEY=your_key_here
```



## Run

Open the notebook and run the cells one at a time, in order. Do not use "Run all", because it can trigger `429` rate-limit errors. Running the notebook calls the Gemini API and may use credits.

To try the agent interactively without the notebook, from the repository root:

```bash
adk web 
```

Then open `http://localhost:8000` and select `relapse_agent`.

To run the ADK evaluation:

```bash
adk eval relapse_agent relapse_agent/integration.evalset.json --config_file_path=relapse_agent/test_config.json --print_detailed_results
```

## Deployment (optional, not run)

The notebook prepares `relapse_agent_deploy/` and contains the `adk deploy agent_engine` command. To use it you need a Google Cloud project with billing and the Vertex AI APIs enabled. Set `RUN_DEPLOYMENT = True` and `PROJECT_ID` in Section 12. Delete the deployed agent afterwards (Section 12.5) to avoid cost.

## Limitations

- The test set is small and invented.
- The tool uses simple word patterns for death, relapse and negation, so unusual phrasing can be missed.
- A translation error that drops a negation would change the answer.
- Use only de-identified text. Do not send real patient data to a model or cloud service without approval from your organisation.
- The deployment was not run.


