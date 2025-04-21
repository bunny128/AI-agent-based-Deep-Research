# Deep Research AI Agent

This project is a modular and scalable AI research assistant designed with a dual-agent architecture. It uses the LangChain and LangGraph frameworks to orchestrate the agents, and Tavily API, Wikipedia, and arXiv for real-time knowledge gathering. The system can take any user query, autonomously conduct online research, and generate a well-drafted response — all powered by AI.

## Problem Statement
Design a Deep Research AI Agentic System that crawls websites using Tavily for online information gathering. Implement a dual-agent system with:
* One agent focused on research and data collection using multiple tools.
* Another agent functioning as an answer drafter.
The system should utilize the LangGraph & LangChain frameworks to effectively organize the gathered information.


## Features

- **Arxiv Search:** Queries academic papers from Arxiv.
- **Wikipedia Search:** Queries Wikipedia articles for general knowledge.
- **Tavily Search:** Performs general web search using Tavily API.
- **LLM-based Answering:** Utilizes the Groq model to process and summarize research results.
- **Interactive Interface:** Easy-to-use Streamlit UI for entering queries and viewing results.
- **Custom Graph Architecture:** Built with LangGraph to execute tool interactions and provide accurate, concise answers.

## Requirements

To run this app, you need the following Python libraries:

- Streamlit
- dotenv
- langchain_community
- langchain_groq
- langchain_core
- langgraph
- requests
- typing_extensions

## Usage

- Enter your search query in the provided text area.
- Enter your API keys in the sidebar configuration.
- Click "Search" to query the tools and get a summary from the language model.
- View the clean, processed answer in the Final Answer section.
- Optionally, view the raw outputs from each tool in the Raw Tool Outputs section.

## Workflow

- **User Input:** The user provides a search query and API keys.
- **Tool Execution:** The system queries the selected tools (Arxiv, Wikipedia, Tavily) using LangChain API wrappers.
- **LLM Processing:** The results are processed by the Groq model, which returns a concise and factual response.
- **State Management:** LangGraph manages the state flow, ensuring that the results are collected, processed, and displayed in an organized manner.
- **Final Answer Display:** The cleaned-up answer is shown to the user, along with the option to view raw tool outputs.

## Error Handling

If the user doesn't provide the necessary API keys or query, a warning message will be displayed. In case of any technical error during the process, a general error message will be shown.

## Challenges in Building
* Ensuring accuracy and relevancy of web-crawled content
* Managing asynchronous API calls in a structured flow
* Multi-source aggregation of data
* Maintaining coherence across multi-turn agent processing
* Avoiding hallucination in drafted outputs
* Clean integration of LangChain tools with LangGraph logic

## Future Enhancements
* Feedback loops to verify and refine drafts
* Multi-agent branching based on task type
* Memory integration for persistent context
* Rich Streamlit UI with output logs and source links

## Example Use Case

- **Query:** "What is the theory of relativity?"
  
The system will fetch relevant articles from Arxiv, Wikipedia, and Tavily, summarize the findings, and provide a concise answer.

## Contributing

If you'd like to contribute, feel free to open an issue or submit a pull request.

## License

This project is open-source and available under the MIT License.

## Credits
Built by Vaibhav Jakhar & Tony (AI Assistant) — a powerful duo working on practical AGI tools.


