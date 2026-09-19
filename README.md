# Agentic RAG System with CrewAI

A multi-agent Retrieval-Augmented Generation (RAG) system built with CrewAI. The system uses a Router Agent to classify user questions and a Retriever Agent to retrieve relevant information from a research-paper PDF or answer general questions using an OpenAI language model.

## Features

- Router Agent for query classification
- Retriever Agent for grounded responses
- PDF search using `PDFSearchTool`
- FAISS-compatible embedding support
- OpenAI chat model integration
- Structured execution logging
- Routing-performance visualization
- Environment-variable-based API-key management
- Error handling for missing files, credentials, and log data

## System Architecture

```text
User Question
      |
      v
Router Agent
      |
      +------------------+------------------+
      |                  |                  |
      v                  v                  v
   PDF Route          Web Route         LLM Route
      |                  |                  |
      v                  v                  v
 PDFSearchTool       Web Search       OpenAI LLM
      |                  |                  |
      +------------------+------------------+
                         |
                         v
                 Retriever Agent
                         |
                         v
                  Final Answer
Agents
Router Agent
The Router Agent classifies each user query into one of the following routes:

PDF: Questions answered from the research paper
WEB: Questions requiring external web search
LLM: General questions that do not require document retrieval
Retriever Agent
The Retriever Agent uses the route selected by the Router Agent and retrieves relevant information using the appropriate tool. It then produces a grounded response.

Requirements
Python 3.10 or later
An OpenAI API key
The research-paper PDF file
Internet access for OpenAI API calls and optional web search
Installation
Install the required packages:

bash

Collapse


 Copy

pip install -U crewai crewai-tools openai langchain-openai langchain-community faiss-cpu python-dotenv pandas matplotlib
If you are using a Jupyter Notebook, install packages in a notebook cell:

python

Collapse


 Copy

%pip install -U crewai crewai-tools openai langchain-openai langchain-community faiss-cpu python-dotenv pandas matplotlib
After installing packages in Jupyter, restart the kernel before continuing.

Project Structure
text

Collapse


 Copy

project/
│
├── transformer_research_paper-dataset.pdf
├── .env
├── README.md
└── agentic_rag.ipynb
Environment Configuration
Create a .env file in the project directory:

env

Collapse


 Copy

OPENAI_API_KEY=your_openai_api_key
Do not commit .env to source control. Add it to .gitignore:

gitignore

Collapse


 Copy

.env
Load the environment variables:

python

Collapse


 Copy

import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY is missing. Add it to the .env file and restart the kernel."
    )
Creating the CrewAI Language Model
The project uses the standard OpenAI API rather than Azure OpenAI or Azure AI Inference:

python

Collapse


 Copy

from crewai import LLM

crewai_llm = LLM(
    model="openai/gpt-4o-mini",
    api_key=OPENAI_API_KEY,
    temperature=0,
)
The crewai_llm object must be created before defining the agents.

PDF Configuration
Verify that the PDF exists:

python

Collapse


 Copy

from pathlib import Path

pdf_path = Path("transformer_research_paper-dataset.pdf")

if not pdf_path.exists():
    raise FileNotFoundError(
        f"PDF file was not found: {pdf_path.resolve()}"
    )
Import and initialize the CrewAI PDF search tool:

python

Collapse


 Copy

from crewai_tools import PDFSearchTool

pdf_tool = PDFSearchTool(
    pdf=str(pdf_path)
)
If the installed version of crewai-tools does not accept the pdf argument, try:

python

Collapse


 Copy

pdf_tool = PDFSearchTool()
Check the installed package version and API if this alternative is required.

Optional Embedding Configuration
If the project creates a FAISS vector store independently of PDFSearchTool, use a separate embedding model:

python

Collapse


 Copy

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY,
)
The embedding model is separate from the chat model:

Chat model: gpt-4o-mini
Embedding model: text-embedding-3-small
A chat model such as gpt-4o-mini should not be used as an embedding model.

Agent Definitions
python

Collapse


 Copy

from crewai import Agent

router_agent = Agent(
    role="Router Agent",
    goal=(
        "Classify each user question as PDF, WEB, or LLM "
        "and select the appropriate response route."
    ),
    backstory=(
        "You analyze user questions and determine whether they require "
        "document retrieval, web search, or a general language-model response."
    ),
    llm=crewai_llm,
    verbose=True,
)

retriever_agent = Agent(
    role="Retriever Agent",
    goal=(
        "Retrieve relevant information and provide an accurate, "
        "well-grounded answer."
    ),
    backstory=(
        "You specialize in using retrieval tools and synthesizing "
        "information into clear answers."
    ),
    tools=[pdf_tool],
    llm=crewai_llm,
    verbose=True,
)
Task Definitions
python

Collapse


 Copy

from crewai import Task

routing_task = Task(
    description="""
    Analyze the following user question:

    {question}

    Classify it as exactly one of:
    - PDF: the answer should come from the research-paper PDF
    - WEB: the answer requires external web information
    - LLM: the question can be answered using general knowledge

    Return the selected route and a brief explanation.
    """,
    expected_output=(
        "The selected route, using one of PDF, WEB, or LLM, "
        "with a brief explanation."
    ),
    agent=router_agent,
)

retrieval_task = Task(
    description="""
    Use the routing result to answer the user's question.

    User question:
    {question}

    Routing result:
    {routing_result}

    If the route is PDF, search the research-paper PDF.
    Provide a concise, accurate, and grounded answer.
    Do not invent information that is not supported by the available context.
    """,
    expected_output=(
        "A clear final answer based on the selected route and retrieved context."
    ),
    agent=retriever_agent,
    context=[routing_task],
)
Crew Configuration
python

Collapse


 Copy

from crewai import Crew, Process

crew = Crew(
    agents=[router_agent, retriever_agent],
    tasks=[routing_task, retrieval_task],
    process=Process.sequential,
    verbose=True,
)
The sequential process ensures that the Router Agent completes its task before the Retriever Agent receives the routing result.

Running a Query
python

Collapse


 Copy

import time

execution_log = []

user_question = (
    "What is the main contribution of the research paper?"
)

start_time = time.perf_counter()

result = crew.kickoff(
    inputs={
        "question": user_question
    }
)

elapsed_time = time.perf_counter() - start_time

print(result)
Execution Logging
Use a consistent schema for logging:

python

Collapse


 Copy

execution_log.append({
    "question": user_question,
    "route": "PDF",
    "elapsed_seconds": elapsed_time,
})
Create a DataFrame with a fixed column schema:

python

Collapse


 Copy

import pandas as pd

LOG_COLUMNS = [
    "question",
    "route",
    "elapsed_seconds",
]

log_df = pd.DataFrame(
    execution_log,
    columns=LOG_COLUMNS,
)

display(log_df)
The fixed schema prevents errors such as:

text

Collapse


 Copy

KeyError: "None of [Index(['question', 'route', 'elapsed_seconds'], dtype='object')] are in the [columns]"
For safe handling when no queries have been executed:

python

Collapse


 Copy

if not execution_log:
    print("No execution records are available yet.")

log_df = pd.DataFrame(
    execution_log,
    columns=LOG_COLUMNS,
)

display(log_df)
Inspect the stored records when debugging:

python

Collapse


 Copy

print("Number of records:", len(execution_log))
print("Available columns:", log_df.columns.tolist())
Routing Visualization
python

Collapse


 Copy

import matplotlib.pyplot as plt

if log_df.empty:
    print("No routing data available for visualization.")
else:
    route_counts = log_df["route"].value_counts()

    route_counts.plot(
        kind="bar",
        color="steelblue",
    )

    plt.title("Routing Decisions")
    plt.xlabel("Route")
    plt.ylabel("Number of Queries")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
Common Errors and Solutions
NameError: name 'crewai_llm' is not defined
Create the language-model object before defining the agents:

python

Collapse


 Copy

from crewai import LLM

crewai_llm = LLM(
    model="openai/gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,
)
In a notebook, confirm that the cell defining crewai_llm has been executed.

NameError: name 'PDFSearchTool' is not defined
Install and import the CrewAI tools package:

bash

Collapse


 Copy

pip install -U crewai-tools
Then run:

python

Collapse


 Copy

from crewai_tools import PDFSearchTool
Restart the kernel after installing the package.

FileNotFoundError for the PDF
Ensure that the PDF is in the same directory as the notebook:

python

Collapse


 Copy

from pathlib import Path

pdf_path = Path("transformer_research_paper-dataset.pdf")

if not pdf_path.exists():
    print(pdf_path.resolve())
    raise FileNotFoundError("Required PDF file is missing.")
KeyError when selecting log columns
The logging dictionaries must use these exact keys:

python

Collapse


 Copy

{
    "question": "...",
    "route": "...",
    "elapsed_seconds": 0.0,
}
Do not use inconsistent names such as:

python

Collapse


 Copy

{
    "query": "...",
    "decision": "...",
    "elapsed": 0.0,
}
unless the visualization code is updated to use those names.

OpenAI API connection errors
Check the following:

OPENAI_API_KEY exists in .env.
The .env file is in the current working directory.
The notebook kernel was restarted after configuration changes.
The API key is valid.
The machine has internet access.
The selected model is available to the account.
Test the key independently:

python

Collapse


 Copy

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.responses.create(
    model="gpt-4o-mini",
    input="Respond with the word OK.",
)

print(response.output_text)
Notebook Execution Order
Run the notebook cells in this order:

Install dependencies.
Restart the kernel.
Import packages.
Load environment variables.
Validate OPENAI_API_KEY.
Create crewai_llm.
Verify that the PDF exists.
Import PDFSearchTool.
Initialize pdf_tool.
Create the Router Agent.
Create the Retriever Agent.
Create tasks.
Create the Crew.
Execute a query.
Append results to execution_log.
Display the log.
Generate visualizations.
Security Best Practices
Never hardcode API keys in Python files or notebooks.
Store credentials in .env.
Add .env to .gitignore.
Do not print the complete API key.
Avoid committing private documents to public repositories.
Restrict API-key permissions where possible.
Rotate exposed keys immediately.
Use separate development and production credentials.
Limitations
PDF answers depend on the quality and indexing of the source document.
Web-search functionality requires a configured web-search tool and API key.
OpenAI API usage may incur costs.
Retrieval quality can vary depending on chunking, embeddings, and query wording.
The route recorded in the log must reflect the Router Agent's actual output.
Future Improvements
Add a dedicated web-search tool for the WEB route.
Parse the Router Agent output programmatically.
Add citations and source references to PDF answers.
Store logs in CSV, SQLite, or a monitoring platform.
Add automated tests for routing and retrieval.
Add retry logic for transient API failures.
Use structured output for route classification.
Add a user interface with Streamlit or Gradio
