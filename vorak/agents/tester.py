# vorak-FRAMEWORK/vorak/agents/tester.py

import os
from typing import Dict, Any

try:
    from langchain.agents import AgentExecutor
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.agents import create_tool_calling_agent
    from langchain_community.tools.tavily_search import TavilySearchResults
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # Define a dummy class if langchain is not available to avoid NameError on type hints
    class AgentExecutor:
        pass

from vorak.core.models import AdversarialPrompt, ModelResponse

class AgentTester:
    """
    Handles the setup and execution of tests against LangChain agents.
    """

    def __init__(self):
        """
        Initializes the AgentTester.
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain libraries are not installed. Please run 'pip install langchain langchain-google-genai langchain-community tavily-python' to use agent testing features."
            )
        
        if not os.getenv("GEMINI_API_KEY"):
            raise ValueError("GEMINI_API_KEY must be set in your environment for the agent's LLM.")
        if not os.getenv("TAVILY_API_KEY"):
            raise ValueError("TAVILY_API_KEY must be set in your environment for the agent's search tool.")
            
        self.agent_executor = self._create_sample_agent()
        print("✅ Sample LangChain agent (using Gemini) initialized successfully.")

    def _create_sample_agent(self) -> "AgentExecutor":
        """
        Creates a basic LangChain agent for demonstration and testing.
        This agent uses Google's Gemini model and a Tavily search tool.
        """
        # --- FIX: Explicitly pass the API key to the constructor ---
        api_key = os.getenv("GEMINI_API_KEY")
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key)
        
        # Define the tools the agent can use
        tools = [TavilySearchResults(max_results=1)]
        
        # Define the prompt template for the agent
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("user", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        # Create the agent
        agent = create_tool_calling_agent(llm, tools, prompt)
        
        # Create and return the agent executor
        return AgentExecutor(agent=agent, tools=tools, verbose=True)

    def evaluate_agent(self, prompt: AdversarialPrompt) -> ModelResponse:
        """
        Sends a prompt to the initialized LangChain agent and captures its response.

        Args:
            prompt (AdversarialPrompt): The adversarial prompt to test the agent with.

        Returns:
            ModelResponse: A standardized response object containing the agent's output.
        """
        try:
            # The agent's response is typically in a dictionary with an 'output' key
            response_dict = self.agent_executor.invoke({"input": prompt.prompt_text})
            output_text = response_dict.get("output", "No 'output' key found in agent response.")
            
            return ModelResponse(
                output_text=output_text.strip(),
                prompt_id=prompt.id,
                model_name="langchain-agent/gemini-1.5-flash"
            )
        except Exception as e:
            error_message = f"An error occurred while invoking the agent: {str(e)}"
            print(f"[bold red]{error_message}[/bold red]")
            return ModelResponse(
                output_text="",
                prompt_id=prompt.id,
                model_name="langchain-agent/gemini-1.5-flash",
                error=error_message
            )
