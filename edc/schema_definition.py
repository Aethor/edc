from typing import List, Dict
import edc.utils.llm_utils as llm_utils
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logger = logging.getLogger(__name__)


class SchemaDefiner:
    """The class to handle the first stage: Open Information Extraction"""

    def __init__(
        self,
        model: AutoModelForCausalLM = None,
        tokenizer: AutoTokenizer = None,
        openai_model=None,
    ) -> None:
        assert openai_model is not None or (model is not None and tokenizer is not None)
        self.model = model
        self.tokenizer = tokenizer
        self.openai_model = openai_model

    def define_schema(
        self,
        input_text_str: str,
        extracted_quads_list: List[List[str]],
        few_shot_examples_str: str,
        prompt_template_str: str,
    ) -> Dict[str, str]:
        """Given a piece of text and a list of quadruples extracted
        from it, define each of the relation present
        """

        relations_present = set()
        for t in extracted_quads_list:
            relations_present.add(t[1])

        filled_prompt = prompt_template_str.format_map(
            {
                "text": input_text_str,
                "few_shot_examples": few_shot_examples_str,
                "relations": relations_present,
                "quadruples": extracted_quads_list,
            }
        )
        messages = [{"role": "user", "content": filled_prompt}]

        if self.openai_model is None:
            completion = llm_utils.generate_completion_transformers(
                messages, self.model, self.tokenizer, answer_prepend="Answer: "
            )
        else:
            completion = llm_utils.openai_chat_completion(
                self.openai_model, None, messages
            )

        relation_definition_dict = llm_utils.parse_relation_definition(completion)

        missing_relations = [
            rel for rel in relations_present if rel not in relation_definition_dict
        ]
        if len(missing_relations) != 0:
            logger.debug(
                f"Relations {missing_relations} are missing from the relation definition!"
            )
        return relation_definition_dict
