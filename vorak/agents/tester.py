# vorak/agents/tester.py

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
        api_key = os.getenv("GEMINI_API_KEY")
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key)
        
        tools = [TavilySearchResults(max_results=1)]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("user", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        agent = create_tool_calling_agent(llm, tools, prompt)
        
        return AgentExecutor(agent=agent, tools=tools, verbose=True)

    def evaluate_agent(self, prompt: AdversarialPrompt) -> ModelResponse:
        """
        Sends a prompt to the initialized LangChain agent and captures its response.
        """
        try:
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
