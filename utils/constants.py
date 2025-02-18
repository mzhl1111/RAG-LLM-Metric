from enum import Enum


LLM_RESPONSE = "llm_response"
class RAGBENCH_COL_NAMES(Enum):
    GOLDEN_ANSWER = "response"
    QUESTION = "question"
    CONTEXT = "documents"
    GOLDEN_ANSWER_SENTENCES = "response_sentences"
    CONTEXT_SENTENCES = "documents_sentences"
    GENERATED_ANSWER = "generated_answer"
    KEY_POINTS = "key_points"

