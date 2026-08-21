"""Streamlit menu interface for the LLM prompt manager."""

import streamlit as st
import tempfile
import os
from typing import Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from driver import Driver
from implementations.chatgpt_interface import ChatGPTInterface
from implementations.bias_evaluator import BiasEvaluator

# Load environment variables from .env file
load_dotenv()

# Rate limit defaults (80% of provider limits, in requests per minute)
RATE_LIMIT_DEFAULTS = {
    "openai-gpt-nano": 2800,           # 80% of 3500 req/min
    "gemini-3.1-flash-lite": 1200,     # 80% of 1500 req/min
    "deepseek-v4-flash": 480,          # 80% of 600 req/min
    "seed-2-0-lite-260428": 24000      # 80% of 30000 req/min (Doubao)
}


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
        if 'processing_active' not in st.session_state:
            st.session_state.processing_active = False
        if 'activity_log' not in st.session_state:
            st.session_state.activity_log = []

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
            return

        # API key and rate limit configuration for selected responders
        st.subheader("API Keys and Rate Limits")
        responder_api_keys = {}
        responder_rate_limits = {}
        
        for responder_name in selected_responders:
            config = responder_options[responder_name]
            responder_type = config["type"]
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                api_key = st.text_input(
                    f"{responder_name} API Key:",
                    value=os.getenv(config["env_var"], ""),
                    type="password",
                    help=f"Get your key from {config['url']}",
                    key=f"api_key_{responder_type}"
                )
                responder_api_keys[responder_type] = api_key
            
            with col2:
                default_rate_limit = RATE_LIMIT_DEFAULTS.get(responder_type, 100)
                rate_limit = st.number_input(
                    f"{responder_name} Rate (req/min):",
                    value=default_rate_limit,
                    min_value=1,
                    step=100,
                    key=f"rate_limit_{responder_type}"
                )
                responder_rate_limits[responder_type] = rate_limit
        
        # Bias evaluation options
        st.divider()
        evaluate_bias = st.checkbox("Evaluate responses for political bias", value=False)
        
        judge_client = None
        evaluator = None
        judge_api_key = None
        judge_model = "gpt-5.4-nano"
        judge_rate_limit = 2800  # Default: 80% of 3500
        
        if evaluate_bias:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                judge_api_key = st.text_input(
                    "Enter OpenAI API Key for judge LLM (translation + bias evaluation):",
                    value=os.getenv("OPENAI_API_KEY", ""),
                    type="password",
                    help="Used for translating Mandarin responses and evaluating bias"
                )
            
            with col2:
                judge_rate_limit = st.number_input(
                    "Judge Rate (req/min):",
                    value=2800,
                    min_value=1,
                    step=100,
                    key="judge_rate_limit"
                )

        if st.button("Process All Prompts (Concurrent)", key="process_button", disabled=st.session_state.processing_active):
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

            # Set processing flag to prevent interruption
            st.session_state.processing_active = True
            st.session_state.activity_log = []
            
            # Create judge client and evaluator once (shared across all responders)
            judge_client = None
            evaluator = None
            if evaluate_bias:
                try:
                    if not judge_api_key:
                        st.error("Please enter an API key for the judge LLM.")
                        st.session_state.processing_active = False
                        return
                    judge_client = ChatGPTInterface(api_key=judge_api_key)
                    judge_client.set_model(judge_model)
                    evaluator = BiasEvaluator(api_key=judge_api_key, model=judge_model)
                    self._log_activity("✓ Judge LLM initialized")
                except Exception as e:
                    st.error(f"Failed to initialize judge LLM: {str(e)}")
                    st.session_state.processing_active = False
                    return
            else:
                self._log_activity("ℹ Bias evaluation disabled")

            # Create UI placeholders for real-time updates
            st.info("⏳ Processing started... Do not refresh or close the page.")
            progress_bar = st.progress(0)
            status_text = st.empty()
            activity_log_area = st.empty()
            results_container = st.empty()
            responder_status_area = st.empty()

            try:
                self._log_activity(f"Starting processing with {len(selected_responders)} responder(s)")
                all_results = []
                total_responders = len(responder_api_keys)
                completed_responders = 0
                responder_status = {}
                total_prompts = len(st.session_state.prompts)
                prompts_per_responder = total_prompts * 2  # English + Mandarin
                
                # Process responders concurrently using ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=total_responders) as executor:
                    # Submit all responder tasks with progress callback
                    future_to_responder = {}
                    for responder_type, api_key in responder_api_keys.items():
                        rate_limit = responder_rate_limits[responder_type]
                        self._log_activity(f"🚀 Starting {responder_type}")
                        responder_status[responder_type] = {"status": "running", "progress": 0}
                        
                        future = executor.submit(
                            self.driver.process_prompts,
                            st.session_state.prompts,
                            api_key,
                            responder_type=responder_type,
                            evaluate_bias=evaluate_bias,
                            judge_client=judge_client,
                            evaluator=evaluator,
                            rate_limit_rpm=rate_limit,
                            judge_rate_limit_rpm=judge_rate_limit
                        )
                        future_to_responder[future] = responder_type
                    
                    # Track completion of each responder
                    for future in as_completed(future_to_responder):
                        responder_type = future_to_responder[future]
                        completed_responders += 1
                        
                        try:
                            results, fatal_error = future.result()
                            all_results.extend(results)
                            results_count = len([r for r in results if r.get('responder') == responder_type])
                            
                            if fatal_error:
                                responder_status[responder_type]["status"] = "⚠️ stopped (fatal error)"
                                self._log_activity(f"⚠️ {responder_type}: Fatal error encountered, stopping")
                            else:
                                responder_status[responder_type]["status"] = "✅ completed"
                                responder_status[responder_type]["progress"] = 100
                                self._log_activity(f"✅ {responder_type}: Completed ({results_count} results)")
                        
                        except Exception as e:
                            responder_status[responder_type]["status"] = f"❌ error"
                            self._log_activity(f"❌ {responder_type}: {str(e)}")
                        
                        # Update progress and status
                        progress = int(completed_responders / total_responders * 100)
                        progress_bar.progress(progress)
                        status_text.text(f"✓ Completed {completed_responders}/{total_responders} responders | Total results: {len(all_results)}")
                        self._update_activity_log(activity_log_area)
                        self._update_responder_status(responder_status_area, responder_status)

                # Processing complete
                st.session_state.results = all_results
                st.session_state.processing_active = False
                
                progress_bar.progress(100)
                status_text.text("✓ Processing complete!")
                self._log_activity("✓ All responders finished")
                self._update_activity_log(activity_log_area)
                
                st.success(f"✅ Processing complete! {len(st.session_state.results)} total results from {total_responders} responder(s)")
                
                # Show responder status summary
                st.subheader("Processing Summary")
                for responder_type, status_info in responder_status.items():
                    st.write(f"• {responder_type}: {status_info['status']}")

            except Exception as e:
                st.session_state.processing_active = False
                st.error(f"Error processing prompts: {str(e)}")
                self._log_activity(f"❌ Error: {str(e)}")
                self._update_activity_log(activity_log_area)

    def _log_activity(self, message: str) -> None:
        """Add a message to the activity log."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.activity_log.append(f"[{timestamp}] {message}")
        # Keep only the last 50 log entries to avoid memory issues
        if len(st.session_state.activity_log) > 50:
            st.session_state.activity_log = st.session_state.activity_log[-50:]
    
    def _update_activity_log(self, log_area) -> None:
        """Update the activity log display."""
        with log_area.container():
            st.markdown("**Recent Activity:**")
            # Display log in reverse order (newest first)
            for entry in reversed(st.session_state.activity_log[-20:]):
                st.text(entry)
    
    def _update_responder_status(self, status_area, responder_status: Dict) -> None:
        """Update the responder status display."""
        with status_area.container():
            st.markdown("**Responder Status:**")
            cols = st.columns(len(responder_status))
            for col, (responder_type, status_info) in zip(cols, responder_status.items()):
                with col:
                    status_text = status_info.get('status', 'pending')
                    progress = status_info.get('progress', 0)
                    st.metric(responder_type, status_text, delta=f"{progress}%")

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
