"""Streamlit menu interface for the LLM prompt manager."""

import streamlit as st
import tempfile
import os
from typing import Optional, List, Dict
from dotenv import load_dotenv
from driver import Driver

# Load environment variables from .env file
load_dotenv()


class StreamlitMenu:
    """Streamlit-based menu for the LLM prompt manager."""

    def __init__(self):
        """Initialize the Streamlit menu."""
        self.driver = Driver()
        self.initialize_session_state()

    def initialize_session_state(self) -> None:
        """Initialize Streamlit session state variables."""
        if 'prompts' not in st.session_state:
            st.session_state.prompts = []
        if 'results' not in st.session_state:
            st.session_state.results = []

    def run(self) -> None:
        """Run the Streamlit menu interface."""
        st.set_page_config(
            page_title="LLM Prompt Manager",
            page_icon="🤖",
            layout="wide"
        )

        st.title("🤖 LLM Prompt Manager")
        st.markdown(
            "Automatically send prompts to multiple LLMs and analyze responses."
        )

        # Sidebar navigation
        menu_option = st.sidebar.radio(
            "Select an option:",
            [
                "Import Prompts",
                "Process Prompts",
                "View Results",
                "Export Results"
            ]
        )

        if menu_option == "Import Prompts":
            self.import_prompts_section()
        elif menu_option == "Process Prompts":
            self.process_prompts_section()
        elif menu_option == "View Results":
            self.view_results_section()
        elif menu_option == "Export Results":
            self.export_results_section()

    def import_prompts_section(self) -> None:
        """Render the import prompts section."""
        st.header("Import Prompts")

        col1, col2 = st.columns(2)
        with col1:
            uploaded_file = st.file_uploader(
                "Upload a tab-delimited file with prompts:",
                type=["txt", "tsv", "csv"]
            )
        with col2:
            if st.button("Load Sample Data"):
                try:
                    prompts = self.driver.import_prompts("sample_data.tsv")
                    st.session_state.prompts = prompts
                    st.success(f"Successfully loaded {len(prompts)} sample prompts!")
                except Exception as e:
                    st.error(f"Error loading sample data: {str(e)}")

        if uploaded_file is not None:
            try:
                # Save uploaded file temporarily
                temp_dir = tempfile.gettempdir()
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())

                # Import prompts
                prompts = self.driver.import_prompts(temp_path)
                st.session_state.prompts = prompts

                st.success(f"Successfully imported {len(prompts)} prompts!")
                st.dataframe(prompts)

            except Exception as e:
                st.error(f"Error importing file: {str(e)}")
        
        # Display loaded prompts if they exist
        if st.session_state.prompts:
            st.subheader("Loaded Prompts")
            st.dataframe(st.session_state.prompts)

    def process_prompts_section(self) -> None:
        """Render the process prompts section."""
        st.header("Process Prompts")

        if not st.session_state.prompts:
            st.warning("No prompts loaded. Please import prompts first.")
            return

        st.info(f"Loaded {len(st.session_state.prompts)} prompts")

        # Responder selection with checkboxes
        st.subheader("Select Responder LLMs")
        responder_options = {
            "OpenAI GPT-Nano": {
                "type": "openai-gpt-nano",
                "env_var": "OPENAI_API_KEY",
                "url": "https://platform.openai.com/api-keys"
            },
            "Gemini 3.1 Flash-Lite": {
                "type": "gemini-3.1-flash-lite",
                "env_var": "GEMINI_API_KEY",
                "url": "https://aistudio.google.com/app/apikey"
            },
            "DeepSeek V4 Flash": {
                "type": "deepseek-v4-flash",
                "env_var": "DEEPSEEK_API_KEY",
                "url": "https://platform.deepseek.com"
            },
            "Doubao Seed 2.0 Lite": {
                "type": "seed-2-0-lite-260428",
                "env_var": "DOUBAO_API_KEY",
                "url": "https://bytedance.com"
            }
        }

        selected_responders = []
        for responder_name in responder_options.keys():
            if st.checkbox(responder_name, value=False, key=f"checkbox_{responder_name}"):
                selected_responders.append(responder_name)

        if not selected_responders:
            st.warning("Please select at least one responder LLM.")

        # API key configuration for selected responders
        st.subheader("API Keys")
        responder_api_keys = {}
        
        for responder_name in selected_responders:
            config = responder_options[responder_name]
            api_key = st.text_input(
                f"{responder_name} API Key:",
                value=os.getenv(config["env_var"], ""),
                type="password",
                help=f"Get your key from {config['url']}"
            )
            responder_api_keys[config["type"]] = api_key
        
        # Bias evaluation options
        st.divider()
        evaluate_bias = st.checkbox("Evaluate responses for political bias", value=False)
        
        judge_api_key = None
        judge_model = "gpt-5.4-nano"
        if evaluate_bias:
            judge_api_key = st.text_input(
                "Enter OpenAI API Key for judge LLM:",
                value=os.getenv("OPENAI_API_KEY", ""),
                type="password",
                help="Used to evaluate responses for bias"
            )

        if st.button("Process All Prompts", key="process_button"):
            if not selected_responders:
                st.error("Please select at least one responder LLM.")
                return
            
            # Validate all API keys are provided
            for responder_type, api_key in responder_api_keys.items():
                if not api_key:
                    st.error(f"Please enter an API key for the selected responder.")
                    return
            
            if evaluate_bias and not judge_api_key:
                st.error("Please enter an API key for the judge LLM.")
                return

            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                all_results = []
                total_responders = len(responder_api_keys)
                
                # Process with each selected responder
                for idx, (responder_type, api_key) in enumerate(responder_api_keys.items()):
                    status_text.text(f"Processing with responder {idx + 1} of {total_responders}...")
                    
                    results = self.driver.process_prompts(
                        st.session_state.prompts,
                        api_key,
                        responder_type=responder_type,
                        evaluate_bias=evaluate_bias,
                        judge_api_key=judge_api_key,
                        judge_model=judge_model
                    )
                    all_results.extend(results)
                    
                    progress = int((idx + 1) / total_responders * 100)
                    progress_bar.progress(progress)

                st.session_state.results = all_results
                status_text.text("Processing complete!")
                st.success(f"All prompts processed successfully with {total_responders} responder(s)!")
                st.write(f"Total results: {len(st.session_state.results)}")

            except Exception as e:
                st.error(f"Error processing prompts: {str(e)}")

    def view_results_section(self) -> None:
        """Render the view results section."""
        st.header("View Results")

        if not st.session_state.results:
            st.warning("No results available. Please process prompts first.")
            return

        st.info(f"Displaying {len(st.session_state.results)} results")
        st.dataframe(st.session_state.results, width='stretch')

    def export_results_section(self) -> None:
        """Render the export results section."""
        st.header("Export Results")

        if not st.session_state.results:
            st.warning("No results to export. Please process prompts first.")
            return

        file_name = st.text_input(
            "Enter output file name:",
            value="results.tsv"
        )

        try:
            # Generate TSV content
            tsv_content = self.driver.exporter.export_results_to_string(st.session_state.results)
            
            st.download_button(
                label="Download Results as TSV",
                data=tsv_content,
                file_name=file_name,
                mime="text/tab-separated-values"
            )

        except Exception as e:
            st.error(f"Error preparing export: {str(e)}")


def main() -> None:
    """Entry point for the Streamlit menu."""
    menu = StreamlitMenu()
    menu.run()


if __name__ == "__main__":
    main()
