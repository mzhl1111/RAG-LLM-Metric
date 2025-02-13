from __future__ import annotations # for pervious python version e.g. 3.9
import json
from typing import List, Dict, Union
from evaluator.base_evaluator import RAGEvaluator
from evaluator.prompt_manager import EvaluationType
from sentence_transformers import SentenceTransformer, util

# TODO: add AnswerEquivalenceEvaluatorWithBert
class AnswerEquivalenceEvaluator(RAGEvaluator):
    """
    From https://arxiv.org/abs/2202.07654, Used their definition of answer equivalence to build prompt.
    This method evaluates if the generated answer is equivalent to the reference answer.
    """
    def pre_process(
        self,
        question: str|List[str],
        context: str|List[str],
        answer: str|List[str],
        **kwargs,
    ) -> str:
        assert "golden_answer" in kwargs, "Missing required input: golden_answer"
        golden_answer = kwargs.get("golden_answer")
        assert len(golden_answer) > 0, "golden_answer is empty"
        
        return self.prompt_manager.build_prompt(
            question=question,
            context=context,
            answer=answer,
            eval_type=EvaluationType.ANSWER_EQUIVALENCE,
            golden_answer = golden_answer
        )
        
    def call_llm(self, processed_data: str) -> str:
        # Execute LLM call with constructed prompt
        return self.llm.generate(processed_data)
    
    def post_process(self, llm_response: str) -> Dict[str, float]:
        """Parse JSON response into scores dictionary"""
        try:
            # Clean response and parse JSON
            response_text = llm_response.strip().replace('```json', '').replace('```', '')
            result = json.loads(response_text)
            
            def get_score(result):
                if result['Q1'] == 'no':
                    return 1
                elif result['Q2'] == 'yes':
                    return 1
                return 0
            
            scores = {
                "equivalence" : get_score(result),
                "raw_output": result
            }
            
            return scores
            
        except (json.JSONDecodeError, KeyError) as e:
            return {
                "equivalence": -1,
                "raw_output": response_text,
                'error': str(e)
            }
            
class RefusalAccuracyEvaluator(RAGEvaluator):
    
    def pre_process(self, question, context, answer):
        pass
    
    def call_llm(self, processed_data):
        pass
    
    def post_process(self, llm_response):
        pass
    

    def evaluate(self, question, context, answer):
        prompt1 = self.prompt_manager.build_prompt(
            question = question,
            context = context,
            answer = answer,
            eval_type = EvaluationType.REFUSAL
        )
        
        resp1 = self.llm.generate(prompt1)
        
        prompt2 = self.prompt_manager.build_prompt(
            question = question,
            context = context,
            answer = answer,
            eval_type = EvaluationType.UNDERSPECIFIED_CHECK
        )
        
        resp2 = self.llm.generate(prompt2)
        
        try:
            response_text = resp1.strip().replace('```json', '').replace('```', '')
            result1 = json.loads(response_text)
            
            score1 = {
                "refusal": result1['refusal'],
                "reason": result1['reason']
            }
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing LLM response on refusal: {response_text}")
            score1 = {'refusal': 0xffffffff, "error": str(e)}

        try:
            response_text = resp2.strip().replace('```json', '').replace('```', '')
            result2= json.loads(response_text)
            
            score2 = {
                "underspecifie_check": result2['underspecifie_check'],
                "reason": result2['reason']
            }
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing LLM response on refusal: {response_text}")
            score1 = {'underspecifie_check': 0, "error": str(e)}
            
        return {'refusal_result': score1, "underspecifie_check_score": score2}
            
        

class LearningFacilitationEvaluator(RAGEvaluator):
    def pre_process(
        self,
        question: str|List[str],
        context: str|List[str],
        answer: str|List[str]
    ) -> str:
        return self.prompt_manager.build_prompt(
            question=question,
            context=context,
            answer=answer,
            eval_type=EvaluationType.LEARNING_FACILITATION
        )
        
    def call_llm(self, processed_data: str) -> str:
        # Execute LLM call with constructed prompt
        return self.llm.generate(processed_data)
    
    def post_process(self, llm_response: str) -> Dict[str, float]:
        """Parse JSON response into scores dictionary"""
        try:
            print(f"Raw LLM response: {llm_response}")
            response_text = llm_response.strip().replace('```json', '').replace('```', '')
            result = json.loads(response_text)
            
            scores = {
                "learning_facilitation_score": result['learning_facilitation_score'],
                "educational_strengths": result['educational_strengths'],
                "areas_for_improvement": result['areas_for_improvement'],
                "confidence": result['confidence']
            }
            
            return scores
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing LLM response: {response_text}")
            return {
                "learning_facilitation_score": -1,
                "educational_strengths": [],
                "areas_for_improvement": [],
                "confidence": -1,
                'error': str(e)
            }


class EngagementEvaluator(RAGEvaluator):
    def pre_process(
        self,
        question: Union[str, List[str]],
        context: Union[str, List[str]],
        answer: Union[str, List[str]]
    ) -> str:
        return self.prompt_manager.build_prompt(
            question=question,
            context=context,
            answer=answer,
            eval_type=EvaluationType.ENGAGEMENT_INDEX
        )
        
    def call_llm(self, processed_data: str) -> str:
        # Execute LLM call with constructed prompt
        return self.llm.generate(processed_data)
    
    def post_process(self, llm_response: str) -> Dict[str, Union[float, List[str]]]:
        """Parse JSON response into scores dictionary"""
        try:
            print(f"Raw LLM response: {llm_response}")
            # Clean response and parse JSON
            response_text = llm_response.strip().replace('```json', '').replace('```', '')
            result = json.loads(response_text)
            
            scores = {
                "engagement_score": result.get('engagement_score', -1),
                "engaging_elements": result.get('engaging_elements', []),
                "suggestions_for_improvement": result.get('suggestions_for_improvement', []),
                "confidence": result.get('confidence', -1)
            }
            
            return scores
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing LLM response: {llm_response}")
            return {
                "engagement_score": -1,
                "engaging_elements": [],
                "suggestions_for_improvement": [],
                "confidence": -1,
                'error': str(e)
            }
        

class ContextRelevanceEvaluator(RAGEvaluator):
    """
    From https://arxiv.org/abs/2501.08208, Use their definition of context relevance to build prompt.
    This method evaluates the context relevance of the retrieved context compared to the input question.
    """
    def pre_process(
        self, question: str|List[str], context: str|List[str], answer: str|List[str], **kwargs
    ) -> str:
        return self.prompt_manager.build_prompt(
            question=question,
            context=context,
            eval_type=EvaluationType.CONTEXT_RELEVANCE
        )
        
    def call_llm(self, processed_data: str) -> str:
        # Execute LLM call with constructed prompt
        return self.llm.generate(processed_data)
    
    def post_process(self, llm_response: str) -> Dict[str, float]:
        """Parse JSON response into scores dictionary"""
        try:
            # Clean response and parse JSON
            response_text = llm_response.strip().replace('```json', '').replace('```', '')
            result = json.loads(response_text)
            score = {
                "relevance_score" : result['relevance_score']
            }
            return score
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing LLM response: {response_text}")
            return {
                "fully_answerable": -1,
                'error': str(e)
            }
          

class FactualCorrectnessEvaluator(RAGEvaluator):
    """
    From https://arxiv.org/abs/2407.12873, Use their definition of Factual Correctness to build prompt.
    This method evaluates factual correctness of the generated answer compared to the golden (ground truth) answer.
    """
    def pre_process(
        self, question: str|List[str], context: str|List[str], answer: str|List[str], **kwargs
    ) -> str:
        if "golden_answer" not in kwargs:
            raise KeyError("Missing required key: golden_answer")
        golden_answer = kwargs.get("golden_answer")
        return self.prompt_manager.build_prompt(
            answer=answer,
            eval_type=EvaluationType.FACTUAL_CORRECTNESS,
            golden_answer=golden_answer
        )
        
    def call_llm(self, processed_data: str) -> str:
        # Execute LLM call with constructed prompt
        return self.llm.generate(processed_data)
    
    def post_process(self, llm_response: str) -> Dict[str, float]:
        """Parse JSON response into scores dictionary"""
        try:
            # Clean response and parse JSON
            response_text = llm_response.strip().replace('```json', '').replace('```', '')
            result = json.loads(response_text)
            
            scores = {
                "TP": result['TP'],
                "FP": result['FP'],
                "FN": result['FN'],
                "F1_score": 0 if (result['TP'] + result['FP'] + result['FN']) == 0 else result['TP'] / (result['TP'] + result['FP'] + result['FN']),
            }
            
            return scores
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing LLM response: {response_text}")
            return {
                "TP": -1, "FP": -1, "FN": -1, "F1_SCORE": -1,
                'error': str(e)
            }
        
class AnswerSimilarityEvaluator(RAGEvaluator):
    """
    Computes an embedding-based cosine similarity score between the generated answer and the ground-truth answer.
    Paper:Evaluation of RAG Metrics for Question Answering in the Telecom Domain,https://arxiv.org/abs/2407.12873 
    """
    def __init__(self, llm, prompt_manager, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Args:
            llm: Pass a dummy or None, we won't use it in this evaluator.
            prompt_manager: Not used here, but required by base class signature.
            model_name: The pretrained model name to use for sentence embedding.
        """
        super().__init__(llm, prompt_manager)
        self.model = SentenceTransformer(model_name)

    def pre_process(self, question, context, answer, **kwargs):
        """
        No LLM prompt is needed. Ensure golden_answer is available in kwargs.
        """
        if "golden_answer" not in kwargs:
            raise KeyError("AnswerSimilarityEvaluator requires 'golden_answer' in kwargs.")
        self.generated_answer = answer
        self.golden_answer = kwargs["golden_answer"]
        return None  # We don't need a prompt to pass to an LLM

    def call_llm(self, processed_data: Any) -> str:
        """
        We don't call the LLM here. Return an empty string or dummy response.
        """
        return ""

    def post_process(self, llm_response: str) -> Dict[str, float]:
        """
        Compute the embeddings for the generated answer and the golden answer,return cosine similarity.
        """
        # Encode the answers into sentence embeddings
        gen_emb = self.model.encode(self.generated_answer, convert_to_tensor=True)
        gold_emb = self.model.encode(self.golden_answer, convert_to_tensor=True)

        # Compute cosine similarity
        similarity = util.cos_sim(gen_emb, gold_emb).item()

        return {
            "answer_similarity": float(similarity)  
        }
            
class KeyPointEvaluator(RAGEvaluator):
    """
    From https://arxiv.org/abs/2408.01262, using extracted key points generate from ground truth answer to check with generated answer,
    using the categorized key_points count to calculate generation scores. 
    It can provide completeness, hallucination and irrelevance score. 
    """
    num_key_points = 0
    def pre_process(self, question, context, answer, **kwargs):
        if "key_points" not in kwargs:
            raise KeyError("Missing required input: key_points")
        key_points = kwargs.get("key_points")
        
        if not isinstance(key_points, list):
            raise ValueError("key_points is type of List[str]")
        
        if len(key_points) == 0:
            raise ValueError("key_points is an empty List, which is invalid")
        
        self.num_key_points = len(key_points)
        formatted_key_points = "\n".join(key_points)
        
        return self.prompt_manager.build_prompt(
            question = question,
            answer = answer,
            eval_type = EvaluationType.KEY_POINT,
            key_points = formatted_key_points,
        )
        
    def call_llm(self, processed_data):
        return self.llm.generate(processed_data)
    
    def post_process(self, llm_response):
        try:
            # Clean response and parse JSON
            response_text = llm_response.strip().replace('```json', '').replace('```', '')
            result = json.loads(response_text)
            
            scores = {
                "completeness_score": len(result['complete_ids']) / self.num_key_points,
                "irrelevant_score": len(result['irrelevant_ids']) / self.num_key_points,
                "hallucination_score": len(result['hallucinate_ids']) / self.num_key_points,
                "raw_output" : result
            }
            
            return scores
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing LLM response: {llm_response}")
            return {
                "completeness_score": -1,
                "irrelevant_score": -1,
                "hallucination_score": -1,
                "raw_output" : response_text,
                "error": str(e),
            }
        