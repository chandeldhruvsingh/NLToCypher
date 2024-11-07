import gradio as gr
from langchain_community.llms import HuggingFaceHub
from langchain_community.graphs import Neo4jGraph
from langchain_core.prompts.prompt import PromptTemplate

# Initialize the HuggingFace model and Neo4j connection
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
llm_new = HuggingFaceHub(
    repo_id=repo_id,
    task="text-generation",
    huggingfacehub_api_token="<your-api-key>",  # Replace with your API key
    model_kwargs={
        "max_new_tokens": 100,
        "top_k": 10,
        "top_p": 0.95,
        "typical_p": 0.95,
        "temperature": 0.7,
    },
)

graph = Neo4jGraph(
    url="neo4j+s://46a80122.databases.neo4j.io:7687",
    username="<user>",  # Replace with your username
    password="<pass>",  # Replace with your password
    refresh_schema=True, 
)

schema = graph.schema

CYPHER_GENERATION_TEMPLATE = """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.
Examples: Here are a few examples of generated Cypher statements for particular questions:
# How many published paper are there?
MATCH (a:Article)
RETURN COUNT(a) AS numberOfArticles

The question is:
{question}"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)

def extract_cypher_query(output):
    lines = output.strip().split('\n')
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip().startswith('MATCH'):
            return '\n'.join(lines[i:])
    return None

def execute_cypher_query(graph, cypher_query):
    with graph._driver.session() as session:
        result = session.run(cypher_query)
        return [record.data() for record in result]

def transform_to_natural_language(llm, result, question):
    prompt = f"Transform the following result into a natural language statement:\n\nResult: {result}, where the user question was: {question} \n\nOutput:"
    response = llm.invoke(prompt)
    return response

def get_natural_language_response(nl_query):
    prompt = CYPHER_GENERATION_PROMPT.format(schema=schema, question=nl_query)
    response = llm_new.invoke(prompt)
    
    cypher_query = extract_cypher_query(response)
    
    if cypher_query:
        cypher_result = execute_cypher_query(graph, cypher_query)
        natural_language_response = transform_to_natural_language(llm_new, cypher_result, nl_query)
        return natural_language_response
    else:
        return "Unable to generate a valid Cypher query for the given question."

# Define Gradio interface
def gradio_interface(nl_query):
    return get_natural_language_response(nl_query)

# Set up Gradio UI components
interface = gr.Interface(
    fn=gradio_interface,
    inputs="text",
    outputs="text",
    title="NL to Cypher Query Translator",
    description="Enter a natural language query to retrieve information from the Neo4j database in a human-readable format.",
    examples=[
        ["How many published papers are there?"],
        ["List all authors of published papers."],
        ["What is the total number of articles by Author X?"]
    ],
)

# Launch the Gradio app
if __name__ == "__main__":
    interface.launch()
