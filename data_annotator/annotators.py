import json
from typing import Dict, List

from data_annotator.base_annotator import DataAnnotator
from data_annotator.prompt_manager import AnnotationType, AnnotatePromptManager
from utils.constants import RAGBENCH_COL_NAMES, LLM_RESPONSE, PROMPT
from utils.llm import LLMClient

import numpy as np


class KeyPointAnnotator(DataAnnotator):

    def __init__(
            self,
            llm_class: type[LLMClient] = None,
            **llm_kwargs,
    ):
        super().__init__(llm_class, **llm_kwargs)

    def pre_process(self, row: Dict) -> Dict:
        question = row[RAGBENCH_COL_NAMES.QUESTION.value]
        golden_answer = row[RAGBENCH_COL_NAMES.GOLDEN_ANSWER.value]
        return {
            PROMPT: AnnotatePromptManager().build_prompt(
                question=question,
                golden_answer=golden_answer,
                eval_type=AnnotationType.KEY_POINT_EXTRACTION,
            )
        }

    async def a_call_llm(self, processed: Dict) -> Dict:
        assert processed.get(PROMPT, None), "prompt missing"
        processed[LLM_RESPONSE] = await self.llm.a_generate(prompt=processed[PROMPT])
        return processed

    def post_process(self, processed: Dict, row: Dict) -> Dict:
        try:
            # Clean response and parse JSON
            response_text = processed[LLM_RESPONSE].strip().replace("```json", "").replace("```", "")
            result = json.loads(response_text)
            return {"key_points": result["key_points"]}
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing LLM response for row{row['id']}: {response_text}")
            return {"key_points": ["error"]}


class NumMistakesAnnotator(DataAnnotator):
    def __init__(self, mistakes: List[str]):
        self.mistakes = mistakes
        super().__init__()

    def pre_process(self, row: Dict) -> Dict:
        pass

    async def a_call_llm(self, processed: Dict) -> str:
        pass

    def post_process(self, processed: str, row: Dict) -> Dict:
        np.random.seed(42)
        return {mistake: np.random.choice(3, p=[0.7, 0.2, 0.1]) for mistake in self.mistakes}


class MistakeAnswerGenerator(DataAnnotator):
    def __init__(self, llm_class: type[LLMClient], mistakes: List[str]):
        super().__init__(llm_class=llm_class)
        self.mistakes = mistakes

    def pre_process(self, row: Dict) -> Dict:
        pass

    async def a_call_llm(self, processed: Dict) -> Dict:
        return await self.llm.a_generate(processed=processed[PROMPT])

    def post_process(self, processed: Dict, row: Dict) -> Dict:
        pass


class MistakeAnswerScoringAnnotator(DataAnnotator):
    def __init__(self, scores: List[str]):
        self.scores = scores
        super().__init__()

    def pre_process(self, row: Dict) -> Dict:
        pass

    def a_call_llm(self, processed: Dict) -> Dict:
        pass

    def post_process(self, processed: Dict, row: Dict) -> Dict:
        pass
