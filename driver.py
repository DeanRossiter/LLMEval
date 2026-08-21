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

    def _is_fatal_error(self, error_message: str) -> bool:
        """
        Determine if an error is fatal (should stop processing) or transient (should continue).
        
        Fatal errors: Invalid auth, wrong model, invalid key
        Transient errors: Rate limit, timeout, temporary service issues
        
        Args:
            error_message: The error message from the API.
            
        Returns:
            True if the error is fatal, False if transient.
        """
        fatal_keywords = [
            'invalid api key',
            'unauthorized',
            '401',
            '403',
            'forbidden',
            'model not found',
            '404',
            'invalid model',
            'authentication failed',
            'not authenticated'
        ]
        
        error_lower = error_message.lower()
        return any(keyword in error_lower for keyword in fatal_keywords)

    def _translate_to_english(self, text: str, judge_api_key: str, judge_model: str) -> tuple:
        """
        Translate Mandarin text to English using the judge LLM.

        Args:
            text: Text to translate (in Mandarin).
            judge_api_key: API key for the judge LLM.
            judge_model: Model to use for translation.

        Returns:
            Tuple of (translated_text, error_message). error_message is None if successful.
        """
        try:
            translator = ChatGPTInterface(api_key=judge_api_key)
            translator.set_model(judge_model)
            
            translation_prompt = f"Translate the following Mandarin text to English. Provide only the translation, nothing else:\n\n{text}"
            translated = translator.send_prompt(translation_prompt)
            
            if translated:
                return translated, None
            else:
                return None, "Translation failed: no response from translator"
        except Exception as e:
            error_msg = f"Translation error: {str(e)}"
            return None, error_msg

    def process_prompts(
        self,
        prompts: List[Dict[str, str]],
        responder_api_key: str,
        responder_type: str = "openai-gpt-nano",
        evaluate_bias: bool = False,
        judge_api_key: Optional[str] = None,
        judge_model: str = "gpt-5.4-nano"
    ) -> tuple:
        """
        Process prompts by sending them to an LLM responder and collecting responses.
        Processes each prompt completely (both languages) before moving to the next.
        
        Args:
            prompts: List of prompt dictionaries with 'english_prompt' and 'mandarin_prompt' columns.
            responder_api_key: API key for the responder LLM.
            responder_type: Type of responder ('openai-gpt-nano', 'gemini-3.1-flash-lite', etc).
            evaluate_bias: Whether to evaluate responses for political bias.
            judge_api_key: API key for the judge LLM (required if evaluate_bias is True).
            judge_model: Model to use for bias evaluation.

        Returns:
            Tuple of (results_list, fatal_error_occurred).
            fatal_error_occurred is True if processing should stop for this responder.
        """
        # Initialize AI interface based on responder type
        try:
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
        except Exception as e:
            error_msg = f"Failed to initialize responder: {str(e)}"
            print(f"FATAL ERROR: {error_msg}")
            return [], True

        # Initialize bias evaluator if requested
        evaluator = None
        if evaluate_bias and judge_api_key:
            try:
                evaluator = BiasEvaluator(api_key=judge_api_key, model=judge_model)
            except Exception as e:
                print(f"Warning: Could not initialize evaluator: {str(e)}")

        results = []
        fatal_error_occurred = False

        # Process each prompt completely (English + Mandarin) before moving to next
        for prompt_idx, prompt_dict in enumerate(prompts):
            # Get English and Mandarin prompts
            english_prompt = ''
            mandarin_prompt = ''
            
            for key, value in prompt_dict.items():
                key_lower = key.lower()
                if key_lower == 'english_prompt':
                    english_prompt = value
                elif key_lower == 'mandarin_prompt':
                    mandarin_prompt = value

            # ===== ENGLISH PROMPT =====
            if english_prompt:
                english_response = self.ai_interface.send_prompt(english_prompt)
                english_error = None
                english_status = "success"
                
                if not english_response:
                    english_status = "error_response"
                    english_error = "No response received (check API key and available credits)"
                    
                    # Check if this is a fatal error
                    if self._is_fatal_error(english_error):
                        fatal_error_occurred = True
                        print(f"FATAL ERROR at row {prompt_idx}: {english_error}")
                
                # Create English result
                english_result = {
                    **prompt_dict,
                    'language': 'English',
                    'response': english_response,
                    'response_translated': None,
                    'status': english_status,
                    'error': english_error,
                    'responder': responder_type,
                    'factual_balance': None,
                    'framing_bias': None,
                    'attribution_of_responsibility': None,
                    'political_avoidance': None,
                    'loaded_language': None,
                    'bias_justification': None
                }
                
                # Evaluate English response if successful and evaluator is available
                if english_response and evaluator:
                    try:
                        evaluation = evaluator.evaluate(english_prompt, english_response)
                        if evaluation:
                            english_result.update({
                                'factual_balance': evaluation.get('factual_balance'),
                                'framing_bias': evaluation.get('framing_bias'),
                                'attribution_of_responsibility': evaluation.get('attribution_of_responsibility'),
                                'political_avoidance': evaluation.get('political_avoidance'),
                                'loaded_language': evaluation.get('loaded_language'),
                                'bias_justification': evaluation.get('justification')
                            })
                    except Exception as e:
                        english_result['status'] = 'error_evaluation'
                        english_result['error'] = f"Evaluation failed: {str(e)}"
                
                results.append(english_result)

            # ===== MANDARIN PROMPT =====
            if mandarin_prompt:
                mandarin_response = self.ai_interface.send_prompt(mandarin_prompt)
                mandarin_error = None
                mandarin_status = "success"
                translated_response = None
                
                if not mandarin_response:
                    mandarin_status = "error_response"
                    mandarin_error = "No response received (check API key and available credits)"
                    
                    # Check if this is a fatal error
                    if self._is_fatal_error(mandarin_error):
                        fatal_error_occurred = True
                        print(f"FATAL ERROR at row {prompt_idx}: {mandarin_error}")
                
                # Translate Mandarin response if successful
                if mandarin_response and judge_api_key:
                    translated_response, translation_error = self._translate_to_english(
                        mandarin_response, judge_api_key, judge_model
                    )
                    
                    if translation_error:
                        mandarin_status = "error_translation"
                        mandarin_error = translation_error
                
                # Create Mandarin result
                mandarin_result = {
                    **prompt_dict,
                    'language': 'Mandarin',
                    'response': mandarin_response,
                    'response_translated': translated_response,
                    'status': mandarin_status,
                    'error': mandarin_error,
                    'responder': responder_type,
                    'factual_balance': None,
                    'framing_bias': None,
                    'attribution_of_responsibility': None,
                    'political_avoidance': None,
                    'loaded_language': None,
                    'bias_justification': None
                }
                
                # Evaluate translated response if translation was successful and evaluator is available
                if translated_response and evaluator:
                    try:
                        evaluation = evaluator.evaluate(mandarin_prompt, translated_response)
                        if evaluation:
                            mandarin_result.update({
                                'factual_balance': evaluation.get('factual_balance'),
                                'framing_bias': evaluation.get('framing_bias'),
                                'attribution_of_responsibility': evaluation.get('attribution_of_responsibility'),
                                'political_avoidance': evaluation.get('political_avoidance'),
                                'loaded_language': evaluation.get('loaded_language'),
                                'bias_justification': evaluation.get('justification')
                            })
                    except Exception as e:
                        mandarin_result['status'] = 'error_evaluation'
                        mandarin_result['error'] = f"Evaluation failed: {str(e)}"
                
                results.append(mandarin_result)
            
            # Stop processing if we hit a fatal error
            if fatal_error_occurred:
                print(f"Stopping processing due to fatal error at row {prompt_idx}")
                break

        return results, fatal_error_occurred

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
