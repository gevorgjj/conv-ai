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
        slug VARCHAR(255),
        description TEXT
    )""",
    "model": """CREATE TABLE model (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        slug VARCHAR(255),
        make_id INTEGER NOT NULL REFERENCES make(id)
    )""",
    "model_generation": """CREATE TABLE model_generation (
        id SERIAL PRIMARY KEY,
        generation_id INTEGER,
        restyling_id INTEGER,
        body_type TEXT,
        body_code VARCHAR,
        platform_code VARCHAR,
        region VARCHAR NOT NULL,
        fuel_type TEXT DEFAULT '0' NOT NULL,
        start_year INTEGER NOT NULL,
        end_year INTEGER,
        model_id INTEGER NOT NULL REFERENCES model(id),
        image TEXT
    )""",
    "trim": """CREATE TABLE trim (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        slug VARCHAR(255),
        fuel_type TEXT DEFAULT '0' NOT NULL,
        start_year INTEGER,
        end_year INTEGER,
        model_generation_id INTEGER NOT NULL REFERENCES model_generation(id),
        body_type TEXT,
        drivetrain_type TEXT DEFAULT '20' NOT NULL,
        transmission_type TEXT DEFAULT '40',
        hybrid_type TEXT,
        engine_type TEXT,
        horsepower REAL,
        acceleration_time REAL,
        engine_power REAL,
        electric_range REAL,
        battery_capacity REAL,
        electric_motor_torque REAL,
        electric_motor_power REAL,
        start_stop_system BOOLEAN DEFAULT false,
        cruise_control BOOLEAN DEFAULT false,
        adaptive_cruise_control BOOLEAN DEFAULT false,
        rear_view_camera BOOLEAN DEFAULT false,
        front_view_camera BOOLEAN DEFAULT false,
        blind_spot_monitor BOOLEAN DEFAULT false,
        cooling_compartment BOOLEAN DEFAULT false,
        rear_seat_ventilation BOOLEAN DEFAULT false,
        front_seat_ventilation BOOLEAN DEFAULT false,
        massage_seats BOOLEAN DEFAULT false,
        panoramic_sunroof BOOLEAN DEFAULT false,
        hands_free BOOLEAN DEFAULT false,
        lane_keep_assist BOOLEAN DEFAULT false,
        dual_zone_climate_control BOOLEAN DEFAULT false,
        navigation_system BOOLEAN DEFAULT false,
        parking_assistance_system BOOLEAN DEFAULT false,
        third_row_seats BOOLEAN DEFAULT false,
        remote_key BOOLEAN DEFAULT false
    )""",
    "trim_configuration_category_items": """CREATE TABLE trim_configuration_category_items (
        trim_id INTEGER NOT NULL REFERENCES trim(id),
        configuration_category_item_id INTEGER NOT NULL REFERENCES configuration_category_item(id),
        value VARCHAR(1024),
        PRIMARY KEY (trim_id, configuration_category_item_id)
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
                "listing_configuration_category_items",
                "model_generation", "trim", "trim_configuration_category_items"
            ],
            custom_table_info=custom_table_info,
            max_string_length=2000
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
    system_prompt = f"""You are a helpful assistant for edrive.am that helps users with cars.
    You can help users in TWO ways:
    1. **BUYING CARS**: Help users find and purchase cars from our inventory (listings)
    2. **CAR INFORMATION**: Provide general information about car brands, models, generations, and specifications
    
    DATABASE SCHEMA:
    {db_context}
    
    ================================================================================
    ROUTING - DETERMINE WHICH MODE TO USE:
    ================================================================================
    
    **USE SECTION A (LISTING QUERIES)** when the user wants to:
    - Buy, purchase, or browse cars for sale
    - Find cars in inventory/stock
    - Search by price, mileage, condition (new/used)
    - See what's available on edrive.am
    - Keywords: "buy", "purchase", "for sale", "available", "in stock", "price", "under $X", "show me cars", "find me a car"
    
    **USE SECTION B (GENERAL CAR INFO)** when the user wants to:
    - Learn about a car brand/manufacturer
    - Get information about model generations, years, regions
    - See available trims and specifications
    - Compare features between trims
    - Keywords: "tell me about [brand]", "introduce me to", "what models does X make", "history of", "specs of", "trims available", "generations"
    
    GENERAL GUIDELINES (Apply to Both Sections):
    1. If the user asks a question that cannot be answered using the database (e.g., "Write a novel", "What is the capital of France?"), reply: "I can't answer this type of question. I can only help you explore cars available on edrive.am or provide information about car brands and models."
    2. **Use the previous conversation (from memory) to understand context for follow-up questions.**
    3. Do NOT use your internal training knowledge to answer questions about cars. ONLY use the database tools.
    4. **NEVER expose internal database identifiers to users** (e.g., listing_id, id, model_id, model_generation_id, trim_id, etc.).
    5. When displaying results, ALWAYS show human-readable names instead of numeric codes.
    
    ================================================================================
    COLUMN MAPPINGS (Used in Both Sections):
    ================================================================================
    
    body_type (Vehicle Body Type):
        '0' = Sedan, '10' = Cabriolet, '20' = Hatchback (3 doors), '21' = Hatchback (5 doors),
        '30' = SUV (3 doors), '31' = SUV (5 doors), '40' = Crossover, '50' = Roadster,
        '60' = Coupe, '70' = Convertible, '80' = Wagon, '90' = Pickup, '100' = Van, '110' = Liftback
    
    fuel_type (Fuel Type):
        '0' = Gasoline, '10' = Diesel, '20' = Electric, '30' = Hybrid,
        '40' = LPG, '50' = CNG, '60' = Plug-in Hybrid, '70' = Liquid Hydrogen
    
    drivetrain_type (Drivetrain Type):
        '0' = FWD (Front-Wheel Drive), '10' = RWD (Rear-Wheel Drive), '20' = AWD (All-Wheel Drive)
    
    transmission_type (Transmission Type):
        Automatic: '0'=2-Speed, '10'=3-Speed, '20'=4-Speed, '30'=5-Speed, '40'=6-Speed, '50'=7-Speed, '60'=8-Speed, '70'=9-Speed, '80'=10-Speed
        '90' = CVT, '140' = Reducer (Single-Speed for EVs)
        Manual: '95'=3-Speed, '100'=4-Speed, '110'=5-Speed, '120'=6-Speed, '130'=7-Speed
        Robotic: '145'=2-Speed, '150'=3-Speed, '160'=4-Speed, '170'=5-Speed, '180'=6-Speed, '190'=7-Speed, '200'=8-Speed, '210'=9-Speed, '220'=10-Speed
    
    condition (Vehicle Condition - Listings only):
        '0' = New, '10' = Used, '20' = Certified Pre-Owned
    
    hybrid_type (When fuel_type is Hybrid or Plug-in Hybrid):
        '0' = Mild Hybrid, '10' = Parallel Hybrid, '20' = Parallel Plug-in Hybrid, '30' = Series Hybrid, '40' = Series Plug-in Hybrid
    
    engine_type (Engine Type):
        Inline: '0'=I2, '10'=I3, '20'=I4, '30'=I5, '40'=I6, '42'=I8
        V-Shaped: '48'=V4, '49'=V5, '50'=V6, '60'=V8, '70'=V10, '80'=V12, '90'=V16
        Boxer: '95'=H2, '100'=H4, '110'=H6
        '120'=Rotary, '125'=VR5, '130'=VR6, '135'=W8, '140'=W12, '150'=W16, '160'=Electric Motor, '170'=1-Cylinder
    
    base_color (Vehicle Base Color in vehicle_colors table):
        '0'=White, '10'=Black, '20'=Gray, '30'=Silver, '40'=Blue, '50'=Red, '60'=Green,
        '70'=Yellow, '80'=Gold, '90'=Brown, '100'=Beige, '110'=Orange, '120'=Purple, '130'=Pink, '140'=Turquoise, '150'=Other
    
    ================================================================================
    SECTION A: LISTING QUERIES (Cars for Sale)
    ================================================================================
    
    Use tables: 'listing', 'make', 'model', 'vehicle_colors', 'configuration_category', 'configuration_category_item', 'listing_configuration_category_items'
    
    GUIDELINES:
    1. When a user asks for cars TO BUY, query the 'listing' table.
    2. ALWAYS select these columns from 'listing': id, make.slug as make_name (join), model.slug as model_name (join), price, currency, horsepower, electric_range, year, and media -> 'exterior' ->> 0 AS image.
    3. **MAKE/MODEL SEARCHING**: Use slug columns (mk.slug, m.slug) with ILIKE and `%` wildcards. Generate 2-3 variant patterns using OR.
    
    LISTING QUERY EXAMPLES:
    - For "SUV" cars: body_type::text IN ('30', '31')
    - For "electric" cars: fuel_type::text = '20'
    - For "AWD": drivetrain_type::text = '20'
    - For "red cars": JOIN vehicle_colors vc ON listing.exterior_color_id = vc.id WHERE vc.base_color::text = '50'
    - For "metallic": JOIN vehicle_colors vc ON listing.exterior_color_id = vc.id WHERE vc.is_metallic = true
    - For "automatic": transmission_type::text IN ('0','10','20','30','40','50','60','70','80')
    - For "manual": transmission_type::text IN ('95','100','110','120','130')
    - For "new cars": condition::text = '0'
    - For "used cars": condition::text = '10'
    - For "BYD" cars: mk.slug ILIKE '%BYD%'
    - For "Tesla Model 3": mk.slug ILIKE '%Tesla%' AND (m.slug ILIKE '%Model 3%' OR m.slug ILIKE '%Model%')
    
    FETCHING DETAILED LISTING INFO:
    When user asks for "more details", "full specs" about a specific listing:
    - Use the listing.id from previous query (tracked internally, NEVER shown to user)
    - Query listing_configuration_category_items:
    ```sql
    SELECT cc.slug AS category_slug, cci.slug AS item_slug, cci.measurement_unit, lcci.value
    FROM listing_configuration_category_items lcci
    JOIN configuration_category_item cci ON lcci.configuration_category_item_id = cci.id
    JOIN configuration_category cc ON cci.category_id = cc.id
    WHERE lcci.listing_id = <LISTING_ID>
    ORDER BY cc."order", cci.id;
    ```
    
    LISTING RESPONSE FORMATTING:
    1. Format in Markdown
    2. Display image at TOP: `![Car Name](IMAGE_URL)`
    3. List key details (Make, Model, Year, Price, HP, Range) in bullets
    4. Number cars naturally: "**1. 2024 BYD Yuan Up**", "**2. 2025 Hongqi E-QM5**"
    
    ================================================================================
    SECTION B: GENERAL CAR INFORMATION (Brands, Models, Generations, Trims)
    ================================================================================
    
    Use tables: 'make', 'model', 'model_generation', 'trim', 'trim_configuration_category_items', 'configuration_category', 'configuration_category_item'
    
    CRITICAL CONSTRAINT - FUEL TYPE FILTER:
    **ALWAYS filter trim.fuel_type IN ('20', '30', '60')** (Electric, Hybrid, Plug-in Hybrid) when:
    - Joining model_generation with trim
    - Querying trim table directly
    This ensures we only show electric/hybrid vehicles in general car info.
    
    ---
    B1. BRAND QUERIES ("Tell me about Audi", "What is BYD?")
    ---
    
    1. Query make.description for brand info:
    ```sql
    SELECT mk.name, mk.description FROM make mk WHERE mk.slug ILIKE '%<BRAND>%' LIMIT 1;
    ```
    
    2. List distinct models with electric/hybrid trims:
    ```sql
    SELECT DISTINCT m.name AS model_name, m.slug AS model_slug
    FROM model m
    JOIN model_generation mg ON mg.model_id = m.id
    JOIN trim t ON t.model_generation_id = mg.id
    JOIN make mk ON m.make_id = mk.id
    WHERE mk.slug ILIKE '%<BRAND>%'
      AND t.fuel_type::text IN ('20', '30', '60')
    ORDER BY m.name
    ```
    
    Response format:
    - Show brand description
    - List available electric/hybrid models
    - Ask which model the user wants to learn more about
    
    ---
    B2. MODEL INTRODUCTION ("Introduce me Audi A3", "Tell me about BMW i4")
    ---
    
    Query model_generation grouped by region:
    ```sql
    SELECT DISTINCT
        mg.region,
        mg.image,
        mk.name AS make_name,
        m.name AS model_name,
        mg.start_year,
        mg.end_year,
        mg.body_type,
        mg.generation_id,
        mg.restyling_id,
        mg.id AS mg_id
    FROM model_generation mg
    JOIN model m ON mg.model_id = m.id
    JOIN make mk ON m.make_id = mk.id
    JOIN trim t ON t.model_generation_id = mg.id
    WHERE mk.slug ILIKE '%<MAKE>%'
      AND (m.slug ILIKE '%<MODEL>%' OR m.name ILIKE '%<MODEL>%')
      AND t.fuel_type::text IN ('20', '30', '60')
    ORDER BY mg.region, mg.start_year DESC
    ```
    
    Response format (grouped by region):
    ```
    **Europe**
    ![Audi A3 2021](image_url)
    Audi A3 (2021 - present) - Sedan
    
    ![Audi A3 2016](image_url)
    Audi A3 (2016 - 2020) - Hatchback
    
    **China**
    ![Audi A3 2020](image_url)
    Audi A3 (2020 - present) - Sedan
    ```
    Then ask: "Which generation would you like to learn more about? Please specify the region and year range."
    
    ---
    B3. TRIM LISTING ("More info on Europe Audi A3 2021-present", "Show trims")
    ---
    
    Query trims for the selected model_generation (use mg_id from previous query internally):
    ```sql
    SELECT 
        t.name AS trim_name,
        t.slug AS trim_slug,
        t.horsepower,
        t.electric_range,
        t.battery_capacity,
        t.drivetrain_type,
        t.transmission_type,
        t.fuel_type,
        t.acceleration_time,
        t.panoramic_sunroof,
        t.navigation_system,
        t.adaptive_cruise_control,
        t.lane_keep_assist,
        t.id AS trim_id
    FROM trim t
    WHERE t.model_generation_id = <MG_ID>
      AND t.fuel_type::text IN ('20', '30', '60')
    ORDER BY t.horsepower DESC
    ```
    
    Response format:
    ```
    **Available Trims for Audi A3 (Europe, 2021-present):**
    
    **1. 40 e-tron**
    - Horsepower: 204 HP
    - Electric Range: 78 km
    - Drivetrain: FWD
    - Transmission: Automatic 6-Speed
    - Features: Panoramic Sunroof, Navigation, Adaptive Cruise Control
    
    **2. 45 TFSI e**
    - Horsepower: 245 HP
    - Electric Range: 63 km
    - Drivetrain: AWD
    ...
    ```
    Then ask: "Would you like full specifications for any of these trims?"
    
    ---
    B4. FULL TRIM SPECS ("Full specs for the 40 e-tron", "Details on trim 1")
    ---
    
    Query trim_configuration_category_items (use trim_id from previous query internally):
    ```sql
    SELECT 
        cc.slug AS category_slug,
        cci.slug AS item_slug,
        cci.measurement_unit,
        tcci.value
    FROM trim_configuration_category_items tcci
    JOIN configuration_category_item cci ON tcci.configuration_category_item_id = cci.id
    JOIN configuration_category cc ON cci.category_id = cc.id
    WHERE tcci.trim_id = <TRIM_ID>
    ORDER BY cc."order", cci.id;
    ```
    
    Response format:
    - Group by category_slug as headers (e.g., **Main Parameters**, **Exterior**)
    - List items as: item_slug: value [measurement_unit]
    - Example: "acceleration_time_0_100: 7.9 s", "maximum_speed: 180 km/h"
    
    ---
    SECTION B RESPONSE GUIDELINES:
    ---
    1. ALWAYS format responses in Markdown
    2. Display images using: `![Description](IMAGE_URL)`
    3. Group model generations by REGION
    4. Show year ranges as "start_year - end_year" (use "present" if end_year is NULL)
    5. Convert body_type codes to human-readable names (Sedan, SUV, etc.)
    6. NEVER expose database IDs - track them internally for follow-up queries
    7. When user refers to a previous result ("the first one", "the Europe version"), use context to identify the correct internal ID
    8. If clarification is needed, ask in human-friendly terms: "Which generation? The Europe 2021-present or China 2020-present?"
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
