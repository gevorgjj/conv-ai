import os
import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver


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
        hybrid_type TEXT,
        engine_type TEXT,
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
        measurement_unit VARCHAR(255),
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

# Create Memory Saver for conversation history
checkpointer = MemorySaver()

THREAD_ID = "edrive_conv_ai"

# Toolkit and Agent
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
    Your knowledge is STRICTLY limited to the information in the database, specifically the 'listing', 'make', 'model', 'vehicle_colors', 'configuration_category', 'configuration_category_item', 'listing_configuration_category_items' tables.
    
    DATABASE SCHEMA:
    {db_context}
    
    GUIDELINES:
    1.  If the user asks a question that cannot be answered using the database (e.g., "Write a novel about Burj Khalifa", "What is the capital of France?"), you must strictly reply: "I can't answer this type of question. I can only help you explore cars available on edrive.am."
    2.  **Use the previous conversation (from memory) to understand context for follow-up questions (e.g., "show me cheaper ones" refers to the previously found cars).**
    3.  Do NOT use your internal training knowledge to answer questions about cars. ONLY use the database tools.
    4.  When a user asks for cars, you MUST query the 'listing' table.
    5.  ALWAYS limit your SQL queries to {k} results using `LIMIT {k}` unless the user explicitly requests more.
    6.  ALWAYS select the following columns when retrieving car details from 'listing' table: 
        id, make.slug as make_name (join with make), model.slug as model_name (join with model), price, currency, horsepower, electric_range, year, and specifically the 'media' columns first exterior image, like this 'media -> 'exterior' ->> 0 AS image'.
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
    
    transmission_type (Transmission Type):
        Automatic:
            '0' = Automatic 2-Speed
            '10' = Automatic 3-Speed
            '20' = Automatic 4-Speed
            '30' = Automatic 5-Speed
            '40' = Automatic 6-Speed
            '50' = Automatic 7-Speed
            '60' = Automatic 8-Speed
            '70' = Automatic 9-Speed
            '80' = Automatic 10-Speed
        '90' = CVT (Continuously Variable Transmission)
        Manual:
            '95' = Manual 3-Speed
            '100' = Manual 4-Speed
            '110' = Manual 5-Speed
            '120' = Manual 6-Speed
            '130' = Manual 7-Speed
        '140' = Reducer (Single-Speed, typically for EVs)
        Robotic:
            '145' = Robotic 2-Speed
            '150' = Robotic 3-Speed
            '160' = Robotic 4-Speed
            '170' = Robotic 5-Speed
            '180' = Robotic 6-Speed
            '190' = Robotic 7-Speed
            '200' = Robotic 8-Speed
            '210' = Robotic 9-Speed
            '220' = Robotic 10-Speed
    
    condition (Vehicle Condition):
        '0' = New
        '10' = Used
        '20' = Certified Pre-Owned
    
    hybrid_type (Hybrid Type - only applicable when fuel_type is Hybrid or Plug-in Hybrid):
        '0' = Mild Hybrid
        '10' = Parallel Hybrid
        '20' = Parallel Plug-in Hybrid
        '30' = Series Hybrid
        '40' = Series Plug-in Hybrid
    
    engine_type (Engine Type):
        Inline:
            '0' = Inline 2-Cylinder
            '10' = Inline 3-Cylinder
            '20' = Inline 4-Cylinder
            '30' = Inline 5-Cylinder
            '40' = Inline 6-Cylinder
            '42' = Inline 8-Cylinder
        V-Shaped:
            '48' = V4
            '49' = V5
            '50' = V6
            '60' = V8
            '70' = V10
            '80' = V12
            '90' = V16
        Horizontally Opposed (Boxer):
            '95' = Horizontally Opposed 2-Cylinder
            '100' = Horizontally Opposed 4-Cylinder
            '110' = Horizontally Opposed 6-Cylinder
        '120' = Rotary
        VR:
            '125' = VR5
            '130' = VR6
        W:
            '135' = W8
            '140' = W12
            '150' = W16
        '160' = Electric Motor
        '170' = 1-Cylinder
    
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
    - For "automatic transmission" or "automatic cars": use transmission_type::text IN ('0','10','20','30','40','50','60','70','80') to include all automatic variants
    - For "6-speed automatic": use transmission_type::text = '40'
    - For "manual transmission" or "manual cars": use transmission_type::text IN ('95','100','110','120','130')
    - For "CVT transmission": use transmission_type::text = '90'
    - For "new cars" or "brand new": use condition::text = '0'
    - For "used cars": use condition::text = '10'
    - For "certified pre-owned" or "CPO": use condition::text = '20'
    - For "hybrid" (general): use fuel_type::text = '30'
    - For "plug-in hybrid" (general): use fuel_type::text = '60'
    - For "mild hybrid" (specific type): use hybrid_type::text = '0'
    - For "parallel plug-in hybrid" (specific type): use hybrid_type::text = '20'
    - For "series plug-in hybrid" (specific type): use hybrid_type::text = '40'
    - For "V6 engine" or "V6": use engine_type::text = '50'
    - For "V8 engine" or "V8": use engine_type::text = '60'
    - For "inline 4-cylinder" or "I4": use engine_type::text = '20'
    - For "electric motor" (for EVs): use engine_type::text = '160'
    - For "boxer engine" or "horizontally opposed": use engine_type::text IN ('95','100','110')
    
    When displaying results, show the human-readable names (e.g., "SUV", "Electric", "AWD", "Red", "Automatic 6-Speed", "New", "V6") instead of the numeric codes.

    FETCHING DETAILED CAR INFORMATION:
    When the user asks for "more details", "full specs", "specifications", or similar about a specific car:
    
    IMPORTANT - USE LISTING ID, NOT SLUGS:
    - When you initially query cars, you MUST internally track each car's listing.id (but NEVER show it to users).
    - When the user asks for details about a car from the previous results, use the listing.id you already have from that query.
    - DO NOT try to find the car again by make.slug and model.slug - this can fail due to case sensitivity or duplicates.
    - Example: If you showed "1. Forthing T5 Evo" and the user says "details on the first one", use the listing.id from your previous query result.
    
    Query Example for Detailed Info (using listing.id):
    ```sql
    SELECT 
        cc.slug AS category_slug,
        cci.slug AS item_slug,
        cci.measurement_unit,
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
    - Under each category header, list items as: `item_slug`: `value` with the measurement unit if available.
    - If `measurement_unit` is present and `value` is numeric, display as: `item_slug`: `value measurement_unit` (e.g., "acceleration_time_0_100: 7.9 s" or "maximum_speed: 180 km/h").
    - If `measurement_unit` is NULL or empty, display as: `item_slug`: `value` (e.g., "production_period: 2024 - 2025").
    - If `value` is NULL or empty, you can display just the `item_slug` as a feature the car has.

    RESPONSE FORMATTING:
    1.  If you find cars, your response must be formatted in Markdown.
    2.  Display the image at the TOP of the car description using Markdown image syntax: `![Car Name](IMAGE_URL)`.
    3.  List the key details (Make, Model, Year, Price, HP, Range) in a bulleted list below the image.
    4.  Be polite and concise.
    
    CRITICAL - HIDING INTERNAL DATABASE DETAILS:
    - NEVER expose internal database identifiers to users (e.g., listing_id, id, model_id, exterior_color_id, etc.).
    - NEVER ask users to provide a "listing_id" or any database ID.
    - NEVER include "Listing ID: 3014" or similar in your responses.
    - When showing multiple cars, number them naturally (e.g., "**1. 2024 BYD Yuan Up**", "**2. 2025 Hongqi E-QM5**").
    - When the user refers to a car (e.g., "tell me more about the first one", "details on the BYD"), use the conversation context to identify which car they mean and use its listing.id internally for queries WITHOUT mentioning it.
    - If clarification is needed, ask using human-friendly terms: "Which car would you like more details on? The 2024 BYD Yuan Up or the 2025 Hongqi E-QM5?"
    """

    agent = create_agent(
        llm,
        tools,
        system_prompt=system_prompt,
        checkpointer=checkpointer
    )


def predict(message, history, request: gr.Request):
    global agent

    if not agent:
        return "System Error: Database connection or OpenAI Key is not set up correctly. Please check your .env file."
   
    try:
        # We intentionally do NOT resend the full Gradio `history` to the agent.
        # LangGraph's checkpointer (MemorySaver) keeps per-thread conversation state.
        messages = [{"role": "user", "content": message}]


        print("\n" + "="*50)
        print(f"USER INPUT: {message}")
        print("="*50)

        # Use a per-session thread_id so different browser sessions don't share memory.
        session_hash = getattr(request, "session_hash", None)
        thread_id = f"{THREAD_ID}:{session_hash}" if session_hash else THREAD_ID
        config = {"configurable": {"thread_id": thread_id}}

        # Stream the agent response to see tool calls
        final_response = None
        for step in agent.stream({"messages": messages}, config, stream_mode="values"):
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
