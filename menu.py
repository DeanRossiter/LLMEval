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

        # AI Service selection
        ai_service = st.selectbox(
            "Select AI Service:",
            ["OpenAI (ChatGPT)", "Google (Gemini)"],
            index=0
        )

        # API key configuration
        if ai_service == "OpenAI (ChatGPT)":
            api_key_label = "Enter OpenAI API Key:"
            api_key_help = "Get your key from https://platform.openai.com/api-keys"
            models = ["gpt-3.5-turbo", "gpt-4", "gpt-4o"]
            service_key = "openai"
            default_api_key = st.secrets.get("OPENAI_API_KEY", "")
        else:
            api_key_label = "Enter Google API Key:"
            api_key_help = "Get your key from https://aistudio.google.com/app/apikey"
            models = ["gemini-2.5-flash", "gemini-1.5-pro"]
            service_key = "gemini"
            default_api_key = st.secrets.get("GEMINI_API_KEY", "")

        api_key = st.text_input(
            api_key_label,
            value=default_api_key,
            type="password",
            help=api_key_help
        )

        # Model selection
        model = st.selectbox(
            "Select Model:",
            models
        )

        if st.button("Process All Prompts", key="process_button"):
            if not api_key:
                st.error("Please enter an API key.")
                return

            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                results = self.driver.process_prompts(
                    st.session_state.prompts,
                    api_key,
                    model,
                    ai_service=service_key
                )
                st.session_state.results = results
                progress_bar.progress(100)
                status_text.text("Processing complete!")
                st.success("All prompts processed successfully!")
                st.write("First result sample:")
                st.write(st.session_state.results[0] if st.session_state.results else "No results")
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

        if st.button("Export Results", key="export_button"):
            try:
                self.driver.export_results(st.session_state.results, file_name)
                st.success(f"Results exported to: {file_name}")

            except Exception as e:
                st.error(f"Error exporting results: {str(e)}")


def main() -> None:
    """Entry point for the Streamlit menu."""
    menu = StreamlitMenu()
    menu.run()


if __name__ == "__main__":
    main()
