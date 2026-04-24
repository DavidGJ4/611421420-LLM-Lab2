# 611421420-LLM-Lab2

This version of the README is tailored specifically to your project structure (`my_project`), your directory paths (`src/my_project/crew.py`), and the specific agents you built. It follows your friend's professional structure but removes all emojis as requested.

***

# MyProject Crew: Yelp Star Rating Predictor

Welcome to the MyProject Crew project. This system is a multi-agent AI framework built with crewAI designed to predict Yelp ratings and synthesize reviews by analyzing user history, business features, and market trends.

## Installation

Ensure you have Python >=3.10 <3.14 installed. This project uses uv for dependency management.

First, install uv if you haven't already:
```bash
pip install uv
```

Next, navigate to your project directory and install the dependencies:
```bash
uv sync
```

## Configuration

### LLM Provider
The project is configured to run locally using Ollama with the Phi-3 model. Ensure Ollama is running on your machine.
* Model: ollama/phi3
* URL: http://localhost:11434

### Environment Variables
Create a .env file in the root directory to store your API keys:
```text
SERPER_API_KEY=your_serper_key_here
OTEL_SDK_DISABLED=true
```

## Running the Project

To run the main prediction pipeline:
```bash
uv run my_project
```

This command triggers the CrewAI Flow defined in your project, which coordinates the agents to process the data found in the `data/` directory.

## Project Deliverables and Requirements

This repository fulfills all requested lab requirements:

### 1. Index-Reuse Mechanism Integration
Implemented via the `create_rag_tool` function in `crew.py`. The system performs a check for `chroma.sqlite3` within the `chroma_db` directory. If the index exists, it is reused instantly to avoid unnecessary re-indexing of JSON data, saving time and compute resources.

### 2. Crew with Process.Sequential (Pattern 2)
The `sequential_crew` method implements a collaborative single-task pattern. It ensures that context from the User Profiler flows into the Item Analyst and subsequent agents to arrive at a single, unified prediction.

### 3. Crew with Process.Hierarchical (Manager Agent)
The `hierarchical_crew` method utilizes `Process.hierarchical`. To support local execution, a `manager_llm` is explicitly defined using the local Ollama instance, allowing the manager to delegate tasks to specialists without requiring external API calls.

### 4. New and Stronger Agents
Beyond the basic roles, the following specialist agents were added:
* eda_specialist: Dedicated to performing Exploratory Data Analysis on review datasets to find statistical biases.
* market_researcher: Equipped with SerperDevTool for internet searches to provide real-world context for the businesses being analyzed.
* prediction_modeler: Acts as the final synthesizer to combine all research into a validated Pydantic output.

## Bonus Objectives

### Exploratory Data Analysis (EDA)
The crew generates new knowledge through a dedicated `eda_task`. This task establishes a statistical baseline, such as calculating the user's median star rating and historical rating distribution, which informs the final prediction.

### CrewAI Flow Integration
The project is structured using the `@CrewBase` decorator and integrated into a CrewAI Flow in `main.py`. This allows for stateful management of the crew's execution and easier expansion for complex workflows.

## Directory Structure

* src/my_project/crew.py: Main logic for agents, tools, and crew configurations.
* src/my_project/config/agents.yaml: Configuration file defining agent roles and goals.
* src/my_project/config/tasks.yaml: Configuration file defining task descriptions and expected outputs.
* data/: Contains the source JSON files (filtered_user.json, filtered_item.json, train_review.json).
* chroma_db/: Storage for the persistent vector database indices.

## Current Limitations
When using local models like Phi-3, the quality of the synthesized review depends heavily on the temperature settings and the quantity of data retrieved via RAG. Future iterations will focus on prompt fine-tuning to further reduce the risk of hallucinated business details.
