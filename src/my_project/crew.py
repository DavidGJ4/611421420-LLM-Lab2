import os
import sqlite3
from typing import List
from pydantic import BaseModel, Field
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import JSONSearchTool, SerperDevTool
from langchain_ollama import OllamaLLM

#forcing the local mode...from last lab
os.environ["OPENAI_API_KEY"] = "NA"

class YelpPrediction(BaseModel):
    stars: float = Field(..., description="The predicted star rating")
    review: str = Field(..., description="The synthesized review text")

def create_rag_tool(json_path: str, collection_name: str, config: dict, name: str, description: str) -> JSONSearchTool:
    #looking for the DB...this is the directory i have
    db_file = os.path.join("chroma_db", "chroma.sqlite3")
    
    collection_exists = False
    if os.path.exists(db_file):
        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM collections WHERE name = ?", (collection_name,))
            if cursor.fetchone() is not None:
                collection_exists = True
            conn.close()
        except Exception:
            pass

    if collection_exists:
        #loading instantly from chroma_db folder
        return JSONSearchTool(
            collection_name=collection_name, 
            config=config, 
            embedder_config={"path": "chroma_db"} 
        )
    else:
        #rebuild if not found
        return JSONSearchTool(json_path=json_path, collection_name=collection_name, config=config)

@CrewBase
class MyProjectCrew():
    #defining the LLM and config at the class level or inside a getter
    def __init__(self):
        self.ollama_llm = LLM(
            model="ollama/phi3",
            base_url="http://localhost:11434",
            temperature=0
        )
        self.rag_config = {
            "embedding_model": {
                "provider": "sentence-transformer",
                "config": {"model_name": "BAAI/bge-small-en-v1.5"}
            }
        }
        
    #Tools
    @property
    def user_tool(self):
        return create_rag_tool('data/filtered_user.json', 'v3_hf_user_data', self.rag_config, "search_user_data", "Search user history.")

    @property
    def item_tool(self):
        return create_rag_tool('data/filtered_item.json', 'v3_hf_item_data', self.rag_config, "search_item_data", "Search restaurant features.")

    @property
    def review_tool(self):
        return create_rag_tool('data/train_review.json', 'v3_hf_review_data', self.rag_config, "search_reviews", "Search past review texts.")

    @property
    def web_search_tool(self):
        return SerperDevTool()

    #Agents
    @agent
    def user_profiler(self) -> Agent:
        return Agent(
            config=self.agents_config['user_profiler'],
            llm=self.ollama_llm,
            verbose=True
        )

    @agent
    def item_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['item_analyst'],
            tools=[self.item_tool, self.review_tool],
            llm=self.ollama_llm,
            verbose=True,
            allow_delegation=False
        )

    @agent
    def eda_specialist(self) -> Agent:
        return Agent(config=self.agents_config['eda_specialist'], tools=[self.review_tool], llm=self.ollama_llm, verbose=True)

    @agent
    def market_researcher(self) -> Agent:
        return Agent(config=self.agents_config['market_researcher'], tools=[self.web_search_tool], llm=self.ollama_llm, verbose=True,max_iter=3,allow_delegation=False)

    @agent
    def prediction_modeler(self) -> Agent:
        return Agent(config=self.agents_config['prediction_modeler'], llm=self.ollama_llm, verbose=True)

    #Tasks
    @task
    def analyze_user_task(self) -> Task:
        return Task(config=self.tasks_config['analyze_user_task'])

    @task
    def analyze_item_task(self) -> Task:
        return Task(config=self.tasks_config['analyze_item_task'])

    @task
    def eda_task(self) -> Task: 
        return Task(config=self.tasks_config['eda_task'])

    @task
    def research_task(self) -> Task:
        return Task(config=self.tasks_config['research_task'])

    @task
    def predict_review_task(self) -> Task:
        return Task(config=self.tasks_config['predict_review_task'], output_pydantic=YelpPrediction)

    #Crew Reorganization
    #Pattern 2: Collaborative Single Task (Sequential Process)
    #this fulfills the requirement by having agents work in a specific linear order
    @crew
    def sequential_crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )

    #Pattern 3: Hierarchical Process (Manager Agent)
    #this fulfills the requirement for a Manager-led crew where the Manager LLM
    @crew
    def hierarchical_crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.hierarchical,
            manager_llm=self.ollama_llm,
            verbose=True
        )
        
    @crew
    def crew(self) -> Crew:
        """Default crew called by main.py. Currently set to Hierarchical mode."""
        return self.hierarchical_crew()
