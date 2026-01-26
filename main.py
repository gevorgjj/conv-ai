import os
import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit, create_sql_agent

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Database Connection
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOST", "localhost")
db_port = os.getenv("DB_PORT", "5432")
db_name = os.getenv("DB_NAME", "postgres")

if db_user and db_password and db_host:
    db_uri = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
else:
    # Fallback or error
    db_uri = ""
    print("WARNING: Database credentials not fully found in .env")

print(f"Connecting to database at {db_host}...")

try:
    if db_uri:
        db = SQLDatabase.from_uri(db_uri, 
                                  include_tables=["listing", "make", "model", "vehicle_colors"],
                                  )
        print("Database connection successful.")
    else:
        db = None
        print("No DB URI constructed.")
except Exception as e:
    print(f"Error connecting to database: {e}")
    db = None

# Initialize LLM
if OPENAI_API_KEY:
    llm = ChatOpenAI(temperature=0, model="gpt-5-mini", api_key=OPENAI_API_KEY)
else:
    llm = None
    print("WARNING: OPENAI_API_KEY not found.")

# Toolkit and Agent
agent_executor = None
if db and llm:
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    db_context = db.get_table_info()
    
    # Custom System Message
    system_message = f"""You are a helpful assistant for edrive.am that helps users buy cars.
    Your knowledge is STRICTLY limited to the information in the database, specifically the 'listing' table.
    
    GUIDELINES:
    1.  If the user asks a question that cannot be answered using the database (e.g., "Write a novel about Burj Khalifa", "What is the capital of France?"), you must strictly reply: "I can't answer this type of question. I can only help you explore cars available on edrive.am."
    2.  **Use the provided "Previous conversation" to understand context for follow-up questions (e.g., "show me cheaper ones" refers to the previously found cars).**
    3.  Do NOT use your internal training knowledge to answer questions about cars. ONLY use the database tools.
    4.  When a user asks for cars, you MUST query the 'listing' table.
    5.  ALWAYS select the following columns when retrieving car details: 
        make.slug as make_name (join with make), model.slug as model_name (join with model), price, currency, horsepower, electric_range, year, and specifically the 'media' column.
        (Note: Query the 'slug' column from both the 'make' and 'model' tables using joins).
    
    IMPORTANT COLUMN MAPPINGS:
    The following columns store numeric codes. You MUST use these mappings when querying or displaying data:
    
    body_type (Vehicle Body Type):
        '0' = Sedan
        '10' = Cabriolet
        '20' = Hatchback (3 doors)
        '21' = Hatchback (5 doors)
        '30' = SUV (3 doors)
        '31' = SUV (5 doors)
        '40' = Crossover
        '50' = Roadster
        '60' = Coupe
        '70' = Convertible
        '80' = Wagon
        '90' = Pickup
        '100' = Van
        '110' = Liftback
    
    fuel_type (Fuel Type):
        '0' = Gasoline
        '10' = Diesel
        '20' = Electric
        '30' = Hybrid
        '40' = LPG
        '50' = CNG
        '60' = Plug-in Hybrid
        '70' = Liquid Hydrogen
    
    drivetrain_type (Drivetrain Type):
        '0' = FWD (Front-Wheel Drive)
        '10' = RWD (Rear-Wheel Drive)
        '20' = AWD (All-Wheel Drive)
    
    QUERY EXAMPLES:
    - For "SUV" cars: use body_type::text IN ('30', '31') to include both 3-door and 5-door SUVs
    - For "Hatchback" cars: use body_type::text IN ('20', '21') to include both 3-door and 5-door hatchbacks
    - For "electric" cars: use fuel_type::text = '20'
    - For "AWD" or "all-wheel drive" cars: use drivetrain_type::text = '20'
    
    When displaying results, show the human-readable names (e.g., "SUV", "Electric", "AWD") instead of the numeric codes.

    RESPONSE FORMATTING:
    1.  The 'media' column contains a list of images. You must pick only the first image URL with 'media -> 'exterior' ->> 0 AS image' query.
    2.  If you find cars, your response must be formatted in Markdown.
    3.  Display the image at the TOP of the car description using Markdown image syntax: `![Car Name](IMAGE_URL)`.
    4.  List the key details (Make, Model, Year, Price, HP, Range) in a bulleted list below the image.
    5.  Be polite and concise.
    """

    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=False,
        agent_type="openai-tools",
        agent_executor_kwargs={"handle_parsing_errors": True},
        suffix=system_message
    )

def predict(message, history):
    if not agent_executor:
        return "System Error: Database connection or OpenAI Key is not set up correctly. Please check your .env file."
    
    try:
        # Prepare context from history
        context = ""
        # Handle new Gradio history format (list of dicts)
        for msg in history:
            role = msg.get("role")
            content = msg.get("content")
            
            # Extract text content
            text = ""
            if isinstance(content, list):
                text = " ".join([c.get("text", "") for c in content if c.get("type") == "text"])
            else:
                text = str(content)
            
            if role == "user":
                context += f"Human: {text}\n"
            elif role == "assistant":
                context += f"AI: {text}\n"
        
        # Combine context with current message
        if context:
            full_prompt = f"Previous conversation:\n{context}\n\nCurrent Request: {message}"
            print(f'FULL_PROMPT WITH PREVIOUS CONVERSATION:\n{full_prompt}\n')
        else:
            full_prompt = message

        response = agent_executor.invoke(
            full_prompt
        )
        return response['output']
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Gradio Interface
demo = gr.ChatInterface(
    fn=predict,
    title="🚗 Edrive.am AI Car Assistant",
    description="Ask me about cars in our inventory! I can filter by price, mileage, and more.",
    examples=["Suggest me cars under 20k USD", "Show me electric cars with range over 300km"],
    chatbot=gr.Chatbot(height=700)
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
