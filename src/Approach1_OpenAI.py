import gradio as gr
import yaml
from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI

# Load configurations from config.yaml
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Initialize Neo4j Graph with config values
graph = Neo4jGraph(
    url=config["neo4j"]["url"],
    username=config["neo4j"]["username"],
    password=config["neo4j"]["password"]
)

# Initialize OpenAI LLM with config values
llm = ChatOpenAI(
    temperature=config["openai"]["temperature"],
    openai_api_key=config["openai"]["api_key"]
)

# Create the GraphCypherQAChain
chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True
)

# Define a function to query the knowledge graph
def query_knowledge_graph(message, history):
    try:
        # Get the response from the GraphCypherQAChain
        response = chain.invoke(message)
        
        # Extract the 'result' part of the response
        result = response.get("result", "No result found")
        return str(result)
    except Exception as e:
        return f"An error occurred: {e}"

# Set up and launch the Gradio Chat Interface
gr.ChatInterface(query_knowledge_graph).launch()
