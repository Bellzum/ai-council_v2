"""
AI Council - Streamlit Web Application

Multi-agent consensus orchestration with live visualization.
"""

import asyncio
import streamlit as st
import anthropic

from council.models import CouncilConfig, CouncilAgent, AgentMemory, UploadedDocument
from council.orchestrator import CouncilOrchestrator
from council.document_processor import process_uploaded_file, count_tokens
from council.summarizer import summarize_document


def main():
    st.title("🤖 AI Council v2")
    st.set_page_config(
        page_title="AI Council",
        page_icon="🏛️",
        layout="wide"
    )

    # Initialize session state
    if "page" not in st.session_state:
        st.session_state.page = 1
    if "agents" not in st.session_state:
        st.session_state.agents = []
    if "council_result" not in st.session_state:
        st.session_state.council_result = None
    if "agent_memories" not in st.session_state:
        st.session_state.agent_memories = {}
    if "agent_documents" not in st.session_state:
        st.session_state.agent_documents = {}

    # Sidebar
    st.sidebar.title("🏛️ AI Council")
    st.sidebar.markdown("Multi-agent consensus orchestration")
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Step {st.session_state.page} of 4**")

    # Page routing
    if st.session_state.page == 1:
        page_configure_agents()
    elif st.session_state.page == 2:
        page_council_prompt()
    elif st.session_state.page == 3:
        page_live_consensus()
    elif st.session_state.page == 4:
        page_final_synthesis()


def page_configure_agents():
    """Page 1: Configure council agents."""
    st.header("Step 1: Configure Your Council")
    st.markdown(
        "Define up to 5 agents. The **first agent is the Leader** who creates "
        "and revises the document. The rest are **Evaluators** who provide feedback."
    )

    # API Key input
    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        value=st.session_state.get("api_key", ""),
        help="Your Claude API key (starts with sk-ant-)"
    )
    st.session_state.api_key = api_key

    st.markdown("---")

    # Number of agents selector
    num_agents = st.slider(
        "Number of Agents",
        min_value=2,
        max_value=5,
        value=st.session_state.get("num_agents", 3),
        help="First agent is Leader, rest are Evaluators"
    )
    st.session_state.num_agents = num_agents

    agents = []

    # Default agent configurations
    defaults = [
        ("Author", "Creates and refines the document",
         "You are a skilled writer who creates clear, comprehensive documents. Focus on clarity, structure, and completeness."),
        ("Critic", "Evaluates quality and identifies gaps",
         "You are a critical reviewer who identifies weaknesses, gaps, and areas for improvement. Be thorough but constructive."),
        ("Editor", "Reviews style and coherence",
         "You are an editor focused on style, flow, and readability. Ensure the document is well-organized and engaging."),
        ("Expert", "Provides domain expertise",
         "You are a domain expert who evaluates technical accuracy and completeness. Ensure claims are well-supported."),
        ("Advocate", "Represents the audience perspective",
         "You represent the target audience. Evaluate whether the document meets their needs and is accessible to them."),
    ]

    for i in range(num_agents):
        is_leader = (i == 0)
        role_type = "👑 Leader" if is_leader else f"📋 Evaluator {i}"
        default_name, default_role, default_prompt = defaults[i]

        with st.expander(f"Agent {i+1}: {role_type}", expanded=(i < 2)):
            col1, col2 = st.columns([1, 2])

            with col1:
                name = st.text_input(
                    "Name",
                    value=st.session_state.get(f"agent_{i}_name", default_name),
                    key=f"name_{i}"
                )
                st.session_state[f"agent_{i}_name"] = name

            with col2:
                role_desc = st.text_input(
                    "Role Description",
                    value=st.session_state.get(f"agent_{i}_role", default_role),
                    key=f"role_{i}"
                )
                st.session_state[f"agent_{i}_role"] = role_desc

            starting_prompt = st.text_area(
                "System Prompt",
                value=st.session_state.get(f"agent_{i}_prompt", default_prompt),
                height=100,
                key=f"prompt_{i}"
            )
            st.session_state[f"agent_{i}_prompt"] = starting_prompt

            # Document upload section
            st.markdown("---")
            st.markdown("**Reference Documents** (Optional)")
            memory_key = f"memory_agent_{i}"

            uploaded_files = st.file_uploader(
                "Upload documents for this agent",
                type=['pdf', 'docx', 'txt', 'md'],
                accept_multiple_files=True,
                key=f"upload_{i}",
                help="PDF, Word, or text files that inform this agent's perspective"
            )

            if uploaded_files:
                st.session_state.agent_documents[memory_key] = uploaded_files
                for uf in uploaded_files:
                    st.caption(f"📄 {uf.name} ({uf.size / 1024:.1f} KB)")

            agents.append(CouncilAgent(
                name=name,
                role_description=role_desc,
                starting_prompt=starting_prompt,
                is_leader=is_leader,
                memory_key=memory_key
            ))

    st.session_state.agents = agents

    # Navigation
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Next →", type="primary", use_container_width=True):
            if not api_key or not api_key.startswith("sk-"):
                st.error("Please enter a valid Anthropic API key")
            elif len(agents) < 2:
                st.error("Need at least 2 agents")
            else:
                # Process uploaded documents
                with st.spinner("Processing uploaded documents..."):
                    asyncio.run(process_agent_documents(api_key))
                st.session_state.page = 2
                st.rerun()


async def process_agent_documents(api_key: str):
    """Process all uploaded documents and create summaries."""
    client = anthropic.AsyncAnthropic(api_key=api_key)

    for agent in st.session_state.agents:
        memory_key = agent.memory_key
        uploaded_files = st.session_state.agent_documents.get(memory_key, [])

        if memory_key not in st.session_state.agent_memories:
            st.session_state.agent_memories[memory_key] = AgentMemory(
                agent_name=agent.name
            )

        memory = st.session_state.agent_memories[memory_key]

        for uf in uploaded_files:
            # Check if already processed
            if any(d.filename == uf.name for d in memory.documents):
                continue

            try:
                # Process file
                file_content = uf.read()
                uf.seek(0)  # Reset for potential re-read

                extracted_text, content_type = process_uploaded_file(uf.name, file_content)
                token_count = count_tokens(extracted_text)

                # Generate summary
                summary = await summarize_document(client, extracted_text)

                doc = UploadedDocument(
                    filename=uf.name,
                    content_type=content_type,
                    extracted_text=extracted_text,
                    summary=summary,
                    token_count=token_count
                )
                memory.documents.append(doc)
            except Exception as e:
                st.warning(f"Could not process {uf.name}: {str(e)}")


def page_council_prompt():
    """Page 2: Enter council prompt and configure rounds."""
    st.header("Step 2: Define the Council Task")

    # Council prompt
    council_prompt = st.text_area(
        "Council Prompt",
        value=st.session_state.get("council_prompt", ""),
        height=200,
        placeholder="What should the council deliberate on?\n\nExample: Write a mission statement for a sustainable coffee company that emphasizes quality, environmental responsibility, and community impact.",
        help="This is the main task or question for the council"
    )
    st.session_state.council_prompt = council_prompt

    # Initial context (optional)
    with st.expander("Initial Context (Optional)"):
        initial_context = st.text_area(
            "Starting Document or Context",
            value=st.session_state.get("initial_context", ""),
            height=150,
            help="Provide any starting material the council should build upon"
        )
        st.session_state.initial_context = initial_context

    st.markdown("---")

    # Number of rounds
    max_rounds = st.slider(
        "Maximum Rounds",
        min_value=1,
        max_value=5,
        value=st.session_state.get("max_rounds", 2),
        help="Council will stop early if consensus is reached"
    )
    st.session_state.max_rounds = max_rounds

    # Agent summary
    st.markdown("---")
    st.subheader("Council Members")
    cols = st.columns(len(st.session_state.agents))
    for i, agent in enumerate(st.session_state.agents):
        with cols[i]:
            role = "👑 Leader" if agent.is_leader else "📋 Evaluator"
            st.markdown(f"**{agent.name}**")
            st.caption(f"{role}")
            st.caption(agent.role_description)

    # Navigation
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("← Back", use_container_width=True):
            st.session_state.page = 1
            st.rerun()
    with col3:
        if st.button("Start Council →", type="primary", use_container_width=True):
            if not council_prompt.strip():
                st.error("Please enter a council prompt")
            else:
                st.session_state.page = 3
                st.rerun()


def page_live_consensus():
    """Page 3: Watch consensus unfold with live updates."""
    st.header("Step 3: Council in Session")

    # Build config
    config = CouncilConfig(
        agents=st.session_state.agents,
        council_prompt=st.session_state.council_prompt,
        max_rounds=st.session_state.max_rounds,
        initial_context=st.session_state.get("initial_context", "")
    )

    # Status display
    status_container = st.container()

    # Create columns for document and feedback
    doc_col, feedback_col = st.columns([2, 1])

    with doc_col:
        st.subheader("Current Document")
        document_placeholder = st.empty()

    with feedback_col:
        st.subheader("Evaluator Feedback")
        feedback_placeholder = st.empty()

    # Progress tracking
    current_doc = ""
    feedback_list = []

    def on_round_start(round_num):
        with status_container:
            st.info(f"🔄 Round {round_num}/{config.max_rounds} in progress...")

    def on_leader_response(document):
        nonlocal current_doc
        current_doc = document
        with document_placeholder.container():
            st.markdown(document)

    def on_evaluator_response(agent_name, feedback):
        feedback_list.append(feedback)
        with feedback_placeholder.container():
            for fb in feedback_list:
                icon = "✅" if fb.approved else "⚠️"
                st.markdown(f"{icon} **{fb.agent_name}**")
                if fb.concerns:
                    for c in fb.concerns[:2]:
                        st.caption(f"- {c[:100]}...")
                st.markdown("---")

    def on_round_complete(council_round):
        feedback_list.clear()
        if council_round.all_approved:
            with status_container:
                st.success(f"✅ Consensus reached in round {council_round.round_number}!")

    # Run council
    orchestrator = CouncilOrchestrator(
        config=config,
        api_key=st.session_state.api_key,
        agent_memories=st.session_state.agent_memories,
        on_round_start=on_round_start,
        on_leader_response=on_leader_response,
        on_evaluator_response=on_evaluator_response,
        on_round_complete=on_round_complete,
        verbose=False
    )

    # Execute asynchronously
    try:
        result = asyncio.run(orchestrator.run_council())
        st.session_state.council_result = result

        # Final status
        with status_container:
            if result.consensus_reached:
                st.success(f"✅ Consensus reached in {len(result.rounds)} round(s)!")
            else:
                st.warning("⚠️ Maximum rounds reached without full consensus")

        # Navigation
        st.markdown("---")
        if st.button("View Final Synthesis →", type="primary"):
            st.session_state.page = 4
            st.rerun()

    except Exception as e:
        st.error(f"Error during council execution: {str(e)}")
        if st.button("← Go Back"):
            st.session_state.page = 2
            st.rerun()


def page_final_synthesis():
    """Page 4: Display final synthesized document."""
    st.header("Step 4: Final Synthesis")

    result = st.session_state.council_result

    if not result:
        st.error("No council result available")
        if st.button("Start Over"):
            st.session_state.page = 1
            st.rerun()
        return

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rounds", len(result.rounds))
    col2.metric("Consensus", "✅ Yes" if result.consensus_reached else "⚠️ No")
    col3.metric("Total Tokens", f"{result.total_tokens:,}")
    col4.metric("Duration", f"{result.total_duration_seconds:.1f}s")

    st.markdown("---")

    # Final document
    st.subheader("Final Document")
    st.markdown(result.final_document)

    # Download button
    st.download_button(
        "📥 Download Document",
        result.final_document,
        file_name="council_synthesis.md",
        mime="text/markdown"
    )

    st.markdown("---")

    # Round history
    st.subheader("Deliberation History")
    for council_round in result.rounds:
        with st.expander(
            f"Round {council_round.round_number} - "
            f"{'✅ Consensus' if council_round.all_approved else '🔄 Revised'}"
        ):
            if council_round.leader_reasoning:
                st.markdown("**Leader's Revision Notes:**")
                st.markdown(council_round.leader_reasoning)
                st.markdown("---")

            st.markdown("**Document Version:**")
            st.markdown(council_round.document_version)

            st.markdown("---")
            st.markdown("**Evaluator Feedback:**")
            for fb in council_round.feedback:
                icon = "✅" if fb.approved else "⚠️"
                st.markdown(f"{icon} **{fb.agent_name}**")
                if fb.reasoning:
                    st.caption(fb.reasoning)
                if fb.concerns:
                    st.markdown("Concerns:")
                    for concern in fb.concerns:
                        st.markdown(f"- {concern}")
                if fb.suggestions:
                    st.markdown("Suggestions:")
                    for suggestion in fb.suggestions:
                        st.markdown(f"- {suggestion}")
                st.markdown("---")

    # Start over
    st.markdown("---")
    if st.button("🔄 Start New Council", type="primary"):
        # Clear result but keep agent config
        st.session_state.council_result = None
        st.session_state.council_prompt = ""
        st.session_state.page = 2
        st.rerun()


if __name__ == "__main__":
    main()
