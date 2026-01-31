import os
import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_agent


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

# Custom table info to reduce token usage by excluding unnecessary columns
custom_table_info = {
    "listing": """CREATE TABLE listing (
        id SERIAL PRIMARY KEY,
        mileage INTEGER NOT NULL,
        price INTEGER NOT NULL,
        currency VARCHAR NOT NULL,
        media JSONB,
        trim_name VARCHAR,
        exterior_color_id INTEGER REFERENCES vehicle_colors(id),
        interior_color_id INTEGER REFERENCES vehicle_colors(id),
        model_id INTEGER REFERENCES model(id),
        year INTEGER NOT NULL,
        body_type TEXT NOT NULL,
        fuel_type TEXT NOT NULL,
        condition TEXT,
        horsepower REAL,
        acceleration_time REAL,
        electric_range REAL,
        battery_capacity REAL,
        drivetrain_type TEXT,
        transmission_type TEXT,
        panoramic_sunroof BOOLEAN,
        navigation_system BOOLEAN,
        third_row_seats BOOLEAN
    )""",
    "make": """CREATE TABLE make (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        slug VARCHAR(255)
    )""",
    "model": """CREATE TABLE model (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        slug VARCHAR(255),
        make_id INTEGER NOT NULL REFERENCES make(id)
    )""",
    "vehicle_colors": """CREATE TABLE vehicle_colors (
        id SERIAL PRIMARY KEY,
        name VARCHAR(100) NOT NULL,
        base_color TEXT NOT NULL,
        is_metallic BOOLEAN DEFAULT false,
        is_matte BOOLEAN DEFAULT false,
        hex_code VARCHAR(7) NOT NULL
    )""",
    "configuration_category": """CREATE TABLE configuration_category (
        id SERIAL PRIMARY KEY,
        parent_category_name VARCHAR(255) NOT NULL,
        name VARCHAR(255) NOT NULL,
        slug VARCHAR(255) NOT NULL,
        "order" INTEGER DEFAULT 0
    )""",
    "configuration_category_item": """CREATE TABLE configuration_category_item (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        slug VARCHAR(255) NOT NULL,
        category_id INTEGER REFERENCES configuration_category(id)
    )""",
    "listing_configuration_category_items": """CREATE TABLE listing_configuration_category_items (
        listing_id INTEGER NOT NULL REFERENCES listing(id),
        configuration_category_item_id INTEGER NOT NULL REFERENCES configuration_category_item(id),
        value VARCHAR(1024),
        PRIMARY KEY (listing_id, configuration_category_item_id)
    )"""
}

try:
    if db_uri:
        db = SQLDatabase.from_uri(
            db_uri,
            include_tables=[
                "listing", "make", "model", "vehicle_colors",
                "configuration_category", "configuration_category_item",
                "listing_configuration_category_items"
            ],
            custom_table_info=custom_table_info
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

# print(db.get_table_info())

# Toolkit and Agent
agent_executor = None
if db and llm:
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    all_tools = toolkit.get_tools()
    # query_checker = next(t for t in all_tools if t.name == "sql_db_query_checker")
    query_tool = next(t for t in all_tools if t.name == "sql_db_query")
    tools = [query_tool]
    k = 10
    db_context = db.get_table_info()
    
    # Custom System Message
    system_prompt = f"""You are a helpful assistant for edrive.am that helps users buy cars.
    Your knowledge is STRICTLY limited to the information in the database, specifically the 'listing' table.
    
    DATABASE SCHEMA:
    {db_context}
    
    GUIDELINES:
    1.  If the user asks a question that cannot be answered using the database (e.g., "Write a novel about Burj Khalifa", "What is the capital of France?"), you must strictly reply: "I can't answer this type of question. I can only help you explore cars available on edrive.am."
    2.  **Use the provided "Previous conversation" to understand context for follow-up questions (e.g., "show me cheaper ones" refers to the previously found cars).**
    3.  Do NOT use your internal training knowledge to answer questions about cars. ONLY use the database tools.
    4.  When a user asks for cars, you MUST query the 'listing' table.
    5.  ALWAYS limit your SQL queries to {k} results using `LIMIT {k}` unless the user explicitly requests more.
    6.  ALWAYS select the following columns when retrieving car details: 
        make.slug as make_name (join with make), model.slug as model_name (join with model), price, currency, horsepower, electric_range, year, and specifically the 'media' columns first exterior image, like this 'media -> 'exterior' ->> 0 AS image'.
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
    
    base_color (Vehicle Base Color in vehicle_colors table):
        '0' = White
        '10' = Black
        '20' = Gray
        '30' = Silver
        '40' = Blue
        '50' = Red
        '60' = Green
        '70' = Yellow
        '80' = Gold
        '90' = Brown
        '100' = Beige
        '110' = Orange
        '120' = Purple
        '130' = Pink
        '140' = Turquoise
        '150' = Other
    
    QUERY EXAMPLES:
    - For "SUV" cars: use body_type::text IN ('30', '31') to include both 3-door and 5-door SUVs
    - For "Hatchback" cars: use body_type::text IN ('20', '21') to include both 3-door and 5-door hatchbacks
    - For "electric" cars: use fuel_type::text = '20'
    - For "AWD" or "all-wheel drive" cars: use drivetrain_type::text = '20'
    - For "red cars": JOIN vehicle_colors vc ON listing.exterior_color_id = vc.id WHERE vc.base_color::text = '50'
    - For "black interior": JOIN vehicle_colors vc ON listing.interior_color_id = vc.id WHERE vc.base_color::text = '10'
    - For "metallic cars": JOIN vehicle_colors vc ON listing.exterior_color_id = vc.id WHERE vc.is_metallic = true
    - For "matte cars": JOIN vehicle_colors vc ON listing.exterior_color_id = vc.id WHERE vc.is_matte = true
    - You can combine: "metallic red cars" → WHERE vc.base_color::text = '50' AND vc.is_metallic = true
    
    When displaying results, show the human-readable names (e.g., "SUV", "Electric", "AWD", "Red") instead of the numeric codes.

    FETCHING DETAILED CAR INFORMATION:
    When the user asks for "more details", "full specs", "specifications", or similar about a specific car (identified by listing.id from the conversation context), you MUST query the configuration tables:
    
    Query Example for Detailed Info:
    ```sql
    SELECT 
        cc.slug AS category_slug,
        cci.slug AS item_slug,
        lcci.value
    FROM listing_configuration_category_items lcci
    JOIN configuration_category_item cci ON lcci.configuration_category_item_id = cci.id
    JOIN configuration_category cc ON cci.category_id = cc.id
    WHERE lcci.listing_id = <LISTING_ID>
    ORDER BY cc."order", cci.id;
    ```
    
    DETAILED INFO FORMATTING:
    - Group the configuration items by their `category_slug` (e.g., "Main Parameters", "Exterior", "Emergency Kit").
    - Use the `category_slug` as a header (e.g., **Main Parameters**).
    - Under each category header, list items as: `item_slug`: `value` (e.g., "production_period: 2024 - 2025").
    - If `value` is NULL or empty, you can display just the `item_slug` as a feature the car has.

    RESPONSE FORMATTING:
    1.  If you find cars, your response must be formatted in Markdown.
    2.  Display the image at the TOP of the car description using Markdown image syntax: `![Car Name](IMAGE_URL)`.
    3.  List the key details (Make, Model, Year, Price, HP, Range) in a bulleted list below the image.
    4.  Be polite and concise.
    """

    agent = create_agent(
        llm,
        tools,
        system_prompt=system_prompt,
    )

def predict(message, history):
    if not agent:
        return "System Error: Database connection or OpenAI Key is not set up correctly. Please check your .env file."
   
    try:
        # Build messages list from history
        messages = []
        
        # Handle Gradio history format (list of dicts)
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
                messages.append({"role": "user", "content": text})
            elif role == "assistant":
                messages.append({"role": "assistant", "content": text})
        
        # Add current message
        messages.append({"role": "user", "content": message})

        print("\n" + "="*50)
        print(f"USER INPUT: {message}")
        print("="*50)

        # Stream the agent response to see tool calls
        final_response = None
        for step in agent.stream({"messages": messages}, stream_mode="values"):
            last_message = step["messages"][-1]
            
            # Check if it's a tool call (AIMessage with tool_calls)
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                for tool_call in last_message.tool_calls:
                    print(f"\n🔧 TOOL CALL: {tool_call['name']}")
                    print(f"   Args: {tool_call['args']}")
            
            # Check if it's a tool response
            if hasattr(last_message, 'type') and last_message.type == 'tool':
                print(f"\n📤 TOOL RESPONSE ({last_message.name}):")
                # Truncate long responses for readability
                content = str(last_message.content)
                if len(content) > 500:
                    print(f"   {content[:500]}...")
                else:
                    print(f"   {content}")
            
            final_response = last_message
        
        print("\n" + "="*50)
        print("FINAL RESPONSE:")
        print(final_response.content if hasattr(final_response, 'content') else final_response)
        print("="*50 + "\n")
        
        return final_response.content if hasattr(final_response, 'content') else str(final_response)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"An error occurred: {str(e)}"


demo = gr.ChatInterface(
    fn=predict,
    title="🚗 Edrive.am AI Car Assistant",
    description="Ask me about cars in our inventory! I can filter by price, mileage, and more.",
    examples=["Suggest me cars under 20k USD", "Show me electric cars with range over 300km"],
    chatbot=gr.Chatbot(height=700)
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)


# """
# # logging 
# "question = "show me cars under 100000 usd"

# for step in agent.stream(
#     {"messages": [{"role": "user", "content": question}]},
#     stream_mode="values",
# ):
#     step["messages"][-1].pretty_print()

#     # 1.  The 'media' column contains a list of images. You must pick only the first image URL with 'media -> 'exterior' ->> 0 AS image' query.

# """

# # When querying by color:
# #     - Join vehicle_colors table on the appropriate color_id column
