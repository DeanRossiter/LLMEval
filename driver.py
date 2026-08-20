"""Driver class that orchestrates the LLM prompt manager application."""

from typing import List, Dict, Optional
from implementations.tab_delimited_importer import TabDelimitedImporter
from implementations.chatgpt_interface import ChatGPTInterface
from implementations.gemini_interface import GeminiInterface
from implementations.deepseek_interface import DeepSeekInterface
from implementations.doubao_interface import DoubaoInterface
from implementations.tab_delimited_exporter import TabDelimitedExporter
from implementations.bias_evaluator import BiasEvaluator
from interfaces.ai_interface import AIInterface


class Driver:
    """Main driver class that manages the entire application workflow."""

    def __init__(self):
        """Initialize the driver with implementations."""
        self.importer = TabDelimitedImporter()
        self.exporter = TabDelimitedExporter()
        self.ai_interface: Optional[AIInterface] = None

    def import_prompts(self, file_path: str) -> List[Dict[str, str]]:
        """
        Import prompts from a file.

        Args:
            file_path: Path to the file to import.

        Returns:
            List of prompt dictionaries.
        """
        return self.importer.import_prompts(file_path)

    def _translate_to_english(self, text: str, judge_api_key: str, judge_model: str) -> str:
        """
        Translate Mandarin text to English using the judge LLM.

        Args:
            text: Text to translate (in Mandarin).
            judge_api_key: API key for the judge LLM.
            judge_model: Model to use for translation.

        Returns:
            Translated text in English.
        """
        translator = ChatGPTInterface(api_key=judge_api_key)
        translator.set_model(judge_model)
        
        translation_prompt = f"Translate the following Mandarin text to English. Provide only the translation, nothing else:\n\n{text}"
        translated = translator.send_prompt(translation_prompt)
        
        return translated if translated else text

    def process_prompts(
        self,
        prompts: List[Dict[str, str]],
        responder_api_key: str,
        responder_type: str = "openai-gpt-nano",
        evaluate_bias: bool = False,
        judge_api_key: Optional[str] = None,
        judge_model: str = "gpt-5.4-nano"
    ) -> List[Dict[str, str]]:
        """
        Process prompts by sending them to an LLM responder and collecting responses.
        Handles both English and Mandarin prompts from the same row.

        Args:
            prompts: List of prompt dictionaries with 'prompt_english' and 'prompt_mandarin' columns.
            responder_api_key: API key for the responder LLM.
            responder_type: Type of responder ('openai-gpt-nano', 'gemini-3.1-flash-lite', 'deepseek-v4-flash', 'doubao-seed-2.0-lite').
            evaluate_bias: Whether to evaluate responses for political bias.
            judge_api_key: API key for the judge LLM (required if evaluate_bias is True).
            judge_model: Model to use for bias evaluation.

        Returns:
            List of result dictionaries with one entry per language per row.
        """
        # Initialize AI interface based on responder type
        if responder_type == "openai-gpt-nano":
            self.ai_interface = ChatGPTInterface(api_key=responder_api_key)
            self.ai_interface.set_model("gpt-5.4-nano")
        elif responder_type == "gemini-3.1-flash-lite":
            self.ai_interface = GeminiInterface(api_key=responder_api_key)
            self.ai_interface.set_model("gemini-3.1-flash-lite")
        elif responder_type == "deepseek-v4-flash":
            self.ai_interface = DeepSeekInterface(api_key=responder_api_key)
            self.ai_interface.set_model("deepseek-v4-flash")
        elif responder_type == "seed-2-0-lite-260428":
            self.ai_interface = DoubaoInterface(api_key=responder_api_key)
            self.ai_interface.set_model("seed-2-0-lite-260428")
        else:
            raise ValueError(f"Unknown responder type: {responder_type}")

        # Initialize bias evaluator if requested
        evaluator = None
        if evaluate_bias and judge_api_key:
            evaluator = BiasEvaluator(api_key=judge_api_key, model=judge_model)

        results = []

        for prompt_dict in prompts:
            # Get English and Mandarin prompts from columns (case-insensitive)
            english_prompt = ''
            mandarin_prompt = ''
            
            for key, value in prompt_dict.items():
                key_lower = key.lower()
                if key_lower == 'prompt_english':
                    english_prompt = value
                elif key_lower == 'prompt_mandarin':
                    mandarin_prompt = value

            # Process English prompt
            if english_prompt:
                response = self.ai_interface.send_prompt(english_prompt)
                
                if response:
                    result = {
                        **prompt_dict,
                        'language': 'English',
                        'response': response,
                        'error': None,
                        'responder': responder_type
                    }
                    
                    # Evaluate bias if requested
                    if evaluator:
                        evaluation = evaluator.evaluate(english_prompt, response)
                        if evaluation:
                            result.update({
                                'factual_balance': evaluation.get('factual_balance'),
                                'framing_bias': evaluation.get('framing_bias'),
                                'attribution_of_responsibility': evaluation.get('attribution_of_responsibility'),
                                'political_avoidance': evaluation.get('political_avoidance'),
                                'loaded_language': evaluation.get('loaded_language'),
                                'bias_justification': evaluation.get('justification')
                            })
                else:
                    result = {
                        **prompt_dict,
                        'language': 'English',
                        'response': None,
                        'error': 'No response received (check API key and available credits)',
                        'responder': responder_type
                    }
                
                results.append(result)

            # Process Mandarin prompt
            if mandarin_prompt:
                # Send Mandarin prompt to AI (should get Mandarin response)
                response = self.ai_interface.send_prompt(mandarin_prompt)
                
                if response:
                    # Translate the Mandarin response to English using the judge LLM
                    translated_response = self._translate_to_english(response, judge_api_key, judge_model)
                    
                    result = {
                        **prompt_dict,
                        'language': 'Mandarin',
                        'response': response,
                        'response_translated': translated_response,
                        'error': None,
                        'responder': responder_type
                    }
                    
                    # Evaluate bias on the translated response
                    if evaluator:
                        evaluation = evaluator.evaluate(mandarin_prompt, translated_response)
                        if evaluation:
                            result.update({
                                'factual_balance': evaluation.get('factual_balance'),
                                'framing_bias': evaluation.get('framing_bias'),
                                'attribution_of_responsibility': evaluation.get('attribution_of_responsibility'),
                                'political_avoidance': evaluation.get('political_avoidance'),
                                'loaded_language': evaluation.get('loaded_language'),
                                'bias_justification': evaluation.get('justification')
                            })
                else:
                    result = {
                        **prompt_dict,
                        'language': 'Mandarin',
                        'response': None,
                        'response_translated': None,
                        'error': 'No response received (check API key and available credits)',
                        'responder': responder_type
                    }
                
                results.append(result)

        return results

    def export_results(
        self,
        results: List[Dict[str, str]],
        file_path: str
    ) -> None:
        """
        Export results to a file.

        Args:
            results: List of result dictionaries.
            file_path: Path to export to.
        """
        self.exporter.export_results(file_path, results)

    def run(
        self,
        input_file: str,
        output_file: str,
        api_key: str,
        model: str = "gpt-3.5-turbo"
    ) -> None:
        """
        Run the complete workflow: import, process, and export.

        Args:
            input_file: Path to input file.
            output_file: Path to output file.
            api_key: OpenAI API key.
            model: ChatGPT model to use.
        """
        print(f"Starting LLM Prompt Manager...")
        print(f"Input file: {input_file}")
        print(f"Output file: {output_file}")
        print(f"Model: {model}")

        # Import prompts
        print("\n1. Importing prompts...")
        prompts = self.import_prompts(input_file)
        print(f"   Imported {len(prompts)} prompts")

        # Process prompts
        print("\n2. Processing prompts...")
        results = self.process_prompts(prompts, api_key, model)
        print(f"   Processed {len(results)} prompts")

        # Export results
        print("\n3. Exporting results...")
        self.export_results(results, output_file)

        print("\nWorkflow completed successfully!")
