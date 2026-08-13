"""Driver class that orchestrates the LLM prompt manager application."""

from typing import List, Dict, Optional
from implementations.tab_delimited_importer import TabDelimitedImporter
from implementations.chatgpt_interface import ChatGPTInterface
from implementations.gemini_interface import GeminiInterface
from implementations.tab_delimited_exporter import TabDelimitedExporter
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

    def process_prompts(
        self,
        prompts: List[Dict[str, str]],
        api_key: str,
        model: str = "gpt-3.5-turbo",
        ai_service: str = "openai"
    ) -> List[Dict[str, str]]:
        """
        Process prompts by sending them to an AI service and collecting responses.

        Args:
            prompts: List of prompt dictionaries.
            api_key: API key for the selected service (OpenAI or Google).
            model: Model to use.
            ai_service: AI service to use ('openai' or 'gemini').

        Returns:
            List of result dictionaries with original prompts and responses.
        """
        # Initialize AI interface based on service
        if ai_service.lower() == 'gemini':
            self.ai_interface = GeminiInterface(api_key=api_key)
        else:
            self.ai_interface = ChatGPTInterface(api_key=api_key)
        
        self.ai_interface.set_model(model)

        results = []

        for prompt_dict in prompts:
            # Support both 'prompt' and 'prompts' column names
            prompt_text = prompt_dict.get('prompt', '') or prompt_dict.get('prompts', '')

            if not prompt_text:
                continue

            # Get response from the AI service
            response = self.ai_interface.send_prompt(prompt_text)

            # Create result dictionary with original data plus response
            if response:
                result = {
                    **prompt_dict,
                    'response': response,
                    'error': None,
                    'model': model
                }
            else:
                result = {
                    **prompt_dict,
                    'response': None,
                    'error': 'No response received (check API key and available credits)',
                    'model': model
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
