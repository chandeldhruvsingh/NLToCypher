import yaml
import subprocess
import re
from langchain_community.graphs import Neo4jGraph
import gradio as gr

# Load configurations from config.yaml
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Initialize Neo4j Graph
graph = Neo4jGraph(
    url=config["neo4j"]["url"],
    username=config["neo4j"]["username"],
    password=config["neo4j"]["password"],
    refresh_schema=True
)

# Set paths for model and llama CLI
model_path = config["model"]["path"]
llama_cli_path = config["model"]["llama_cli_path"]

# Function to generate and execute Cypher query from NL query
def process_nl_query(nl_query):
    schema = graph.schema

    # Construct the prompt for the model
    full_prompt = (
        f"<|im_start|>system\nCreate a Cypher statement for the graph with following schema:\n"
        f"{schema}\n\nTo answer the following question:<|im_end|>\n<|im_start|>user\n"
        f"{nl_query}<|im_end|>\n<|im_start|>assistant\n"
    )

    # Command to call llama.cpp CLI
    command = f"{llama_cli_path} -m {model_path} -p '{full_prompt}' --n-predict 180 --temp 0.2 --top_p 0.9"

    # Run the command and capture the output
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error:", result.stderr)
        return f"An error occurred while generating the query: {result.stderr}"

    # Extract the Cypher query from the model's output
    cypher_query = extract_cypher_query(result.stdout)
    if not cypher_query:
        return "Failed to generate a valid Cypher query."

    # Execute the extracted Cypher query on Neo4j
    return execute_cypher_query(graph, cypher_query)

# Function to extract Cypher query from the model output
def extract_cypher_query(text):
    patterns = [
        r"'[\s\n]*(MATCH[\s\S]*?)'",
        r"'''[\s\n]*(MATCH[\s\S]*?)'''",
        r"`[\s\n]*(MATCH[\s\S]*?)`",
        r"```[\s\n]*(MATCH[\s\S]*?)```",
        r"```cypher[\s\n]*(MATCH[\s\S]*?)```"
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[0].strip()
    
    # Fallback pattern to find any Cypher-like query
    cypher_pattern = r"MATCH[\s\S]*?RETURN.*"
    match = re.search(cypher_pattern, text, re.IGNORECASE)
    if match:
        return match.group(0).strip()
    
    return None

# Function to execute Cypher query on Neo4j and return results
def execute_cypher_query(graph, cypher_query):
    with graph._driver.session() as session:
        result = session.run(cypher_query)
        return [record.data() for record in result]

# Gradio interface function
def query_neo4j(nl_query):
    result = process_nl_query(nl_query)
    return result

# Set up Gradio UI
iface = gr.Interface(
    fn=query_neo4j,
    inputs="text",
    outputs="json",
    title="Natural Language to Cypher Query Interface",
    description="Enter a natural language question to query the Neo4j database."
)

# Launch Gradio interface
if __name__ == "__main__":
    iface.launch()
