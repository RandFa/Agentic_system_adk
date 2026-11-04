import os

import vertexai

vertexai.init(
    project=os.environ["GOOGLE_CLOUD_PROJECT"],
    location=os.environ["GOOGLE_CLOUD_LOCATION"],
)

import re

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools import ToolContext
from google.genai import types

retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

MODEL_NAME = "gemini-3.5-flash-lite"

DEATH_PATTERN = re.compile(
    r"\b(died|dead|death|deceased|passed away|succumbed|fatal(?:ly)?)\b", re.IGNORECASE
)
RELAPSE_PATTERN = re.compile(r"\b(relaps\w*|recurr\w*)\b", re.IGNORECASE)
NEGATION_PATTERN = re.compile(
    r"\b(no|not|without|never|free of|absence of|absent)\b", re.IGNORECASE
)


def find_relapse_death_evidence(tool_context: ToolContext) -> dict:
    """Finds the sentences of the translated clinical text that mention death and relapse.

    Reads the English translation that the TranslatorAgent saved in session state
    under the key 'english_text'. Takes no arguments.

    Returns:
        Dictionary with status and evidence.
        Success: {"status": "success", "death_sentences": [...], "relapse_sentences": [...],
                  "sentences_with_both": [...], "relapse_sentences_possibly_negated": [...]}
        Error: {"status": "error", "error_message": "..."}
    """
    text = tool_context.state.get("english_text")
    if not text:
        return {
            "status": "error",
            "error_message": "No English text found in session state under 'english_text'.",
        }

    sentences = re.split(r"(?<=[.!?;])\s+", text.strip())
    death = [s for s in sentences if DEATH_PATTERN.search(s)]
    relapse = [s for s in sentences if RELAPSE_PATTERN.search(s)]
    both = [s for s in death if RELAPSE_PATTERN.search(s)]
    negated = [s for s in relapse if NEGATION_PATTERN.search(s)]

    return {
        "status": "success",
        "death_sentences": death,
        "relapse_sentences": relapse,
        "sentences_with_both": both,
        "relapse_sentences_possibly_negated": negated,
    }


TRANSLATOR_INSTRUCTION = """You are a professional medical translator.
Translate the Original clinical text written by the user in different languages into English. If the text is already in English, return it as is.

Rules:
- Translate faithfully and completely. Do NOT summarize, interpret, add or omit information.
- Keep negations exactly as written (for example 'pas de rechute' -> 'no relapse').
- Use standard English medical terms (for example 'relapse','recurrence','death', 'died').
- Output ONLY the English translation, with no introduction, notes or quotation marks."""

CLASSIFIER_INSTRUCTION = """You are a careful clinical text screener.

The English clinical text to analyse is:
{english_text}

Question: does this text state that the patient died of a relapse (or recurrence) of their disease?

Steps:
1. Call the `find_relapse_death_evidence` tool (it takes no arguments) to get the sentences that mention death and relapse.
2. Decide using the text above, with the tool output as help.

Answer YES only if the text explicitly says the patient died AND attributes the death to a relapse or recurrence.
Answer NO in every other case, including:
- the patient is alive,
- the patient died of another cause,
- the cause of death is not stated,
- a relapse is mentioned but not stated as the cause of death,
- the relapse is negated (for example 'no sign of relapse', 'did not die of a relapse').

Respond with exactly one line and nothing else:
ANSWER: YES
or
ANSWER: NO"""

translator_agent = LlmAgent(
    name="TranslatorAgent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    description="Translates Multilingual clinical text into faithful English.",
    instruction=TRANSLATOR_INSTRUCTION,
    output_key="english_text",
)

classifier_agent = LlmAgent(
    name="RelapseDeathClassifierAgent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    description="Answers whether an English clinical text says the patient died of a relapse.",
    instruction=CLASSIFIER_INSTRUCTION,
    tools=[find_relapse_death_evidence],
    output_key="relapse_verdict",
)

# The ADK CLI tools look for a variable named `root_agent`.
root_agent = SequentialAgent(
    name="RelapseScreeningPipeline",
    description="Translates a Multilingual clinical note and says whether the patient died of a relapse.",
    sub_agents=[translator_agent, classifier_agent],
)
