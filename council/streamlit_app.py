"""
Streamlit entry point for AI Council v2 application.

This file serves as the main entry point for the Streamlit application.
It properly handles imports by adding the project root to the Python path.
"""

import sys
from pathlib import Path

# Add the project root to the Python path so absolute imports work correctly
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Now we can safely import from the council package
import asyncio
import streamlit as st
from council.models import CouncilConfig, CouncilAgent
from council.orchestrator import CouncilOrchestrator


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.council_config = None
        st.session_state.agents = []
        st.session_state.orchestrator = None
        st.session_state.conversation_history = []


def create_sidebar():
    """Create the sidebar with configuration options."""
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # Add configuration controls here
        st.subheader("Council Settings")
        
        # Example configuration
        if st.button("Initialize Council"):
            try:
                st.session_state.council_config = CouncilConfig()
                st.success("Council initialized successfully!")
            except Exception as e:
                st.error(f"Error initializing council: {str(e)}")
        
        st.divider()
        st.subheader("About")
        st.markdown("""
        **AI Council v2**
        
        An intelligent system for multi-agent AI consultation.
        """)


def main():
    """Main Streamlit application function."""
    # Set page configuration
    st.set_page_config(
        page_title="AI Council v2",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Create sidebar
    create_sidebar()
    
    # Main content area
    st.title("🤖 AI Council v2")
    st.markdown("""
    Welcome to the AI Council - a multi-agent collaborative system designed to 
    provide comprehensive insights and decision support through diverse perspectives.
    """)
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Chat", "Agents", "Settings"])
    
    with tab1:
        st.subheader("💬 Conversation")
        
        # Display conversation history
        if st.session_state.conversation_history:
            for message in st.session_state.conversation_history:
                if message["role"] == "user":
                    st.write(f"**You:** {message['content']}")
                else:
                    st.write(f"**Council:** {message['content']}")
        else:
            st.info("Start a conversation by entering your question below.")
        
        # Input area
        st.divider()
        user_input = st.text_area(
            "Enter your question or topic:",
            placeholder="Ask the council about any topic...",
            height=100
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            submit_button = st.button("Submit", type="primary", use_container_width=True)
        
        if submit_button and user_input:
            try:
                # Add user message to history
                st.session_state.conversation_history.append({
                    "role": "user",
                    "content": user_input
                })
                
                # Simulate council response (replace with actual implementation)
                with st.spinner("Council is deliberating..."):
                    # This is where you would call your actual council logic
                    response = f"Thank you for your question: '{user_input}'. The council is processing your query..."
                    
                    st.session_state.conversation_history.append({
                        "role": "assistant",
                        "content": response
                    })
                
                st.success("Response generated!")
                st.rerun()
            except Exception as e:
                st.error(f"Error processing request: {str(e)}")
    
    with tab2:
        st.subheader("🤖 Agents")
        
        if st.session_state.agents:
            for i, agent in enumerate(st.session_state.agents):
                with st.expander(f"Agent {i+1}: {getattr(agent, 'name', 'Unknown')}"):
                    st.write(f"Type: {type(agent).__name__}")
                    st.write(f"Configuration: {agent}")
        else:
            st.info("No agents configured yet. Add agents in the Settings tab.")
    
    with tab3:
        st.subheader("⚙️ Application Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Council Configuration")
            
            # Add fields for council configuration
            st.text_input("Council Name", value="Default Council")
            st.number_input("Number of Agents", min_value=1, max_value=10, value=3)
            st.selectbox(
                "Decision Making Model",
                ["Consensus", "Voting", "Hierarchical", "Collaborative"]
            )
        
        with col2:
            st.markdown("### Logging & Output")
            
            # Add logging options
            st.checkbox("Enable detailed logging")
            st.checkbox("Save conversation history")
            st.checkbox("Export responses as JSON")
        
        # Save configuration button
        if st.button("Save Configuration", type="primary", use_container_width=True):
            st.success("Configuration saved successfully!")
    
    # Footer
    st.divider()
    st.markdown("""
    ---
    **AI Council v2** | Built with Streamlit and Python
    """)


if __name__ == "__main__":
    main()