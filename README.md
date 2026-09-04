# POC for AI insurance paraplanner
Input: A Client Profile - an example of which can be seen from example.json
This format for the client profile is generated from HSBC Insurance's Fact-Finding Form for their clients, essentially a KYC form for insurance. I used tun the his as the format for client profiles.

Output: Recommendations for what Great Eastern insurance products to get

## Architecture
The system is an AI-assisted insurance advisory workflow, where the overall process is controlled by a deterministic LangGraph workflow, while LLM agents are used for tasks that require interpretation and reasoning.
The process starts by taking the user’s raw profile information and normalizing it into a standard client profile. This gives the rest of the system a consistent structure to work with.
Next, the Classifier Agent analyzes the client’s profile to determine which predefined customer archetype best matches them. The agent can use tools to retrieve archetype rules and benchmark information from SQLite, as well as contextual guidance from ChromaDB. However, the agent is not fully trusted: the workflow independently queries the database for valid archetype candidates and checks that the agent’s selected archetype is actually one of those candidates.
After classification, the system performs the financial gap calculation deterministically. The calculate_financial_gaps() function takes the client’s profile and the selected archetype’s benchmark values and calculates things such as the Death/TPD and Critical Illness coverage gaps. This calculation is performed by normal code rather than the LLM, making the financial results predictable and reproducible.
Those calculated gaps are then converted into product requirements. For example, if there is a Death/TPD gap, the system creates a requirement specifying the required coverage amount and relevant product features. This acts as a bridge between the client’s financial needs and the eventual product search.
The system then passes these requirements to the Recommender Agent. The recommender first uses SQLite to identify policies that satisfy the basic metadata requirements, such as product category and client age. It can then use ChromaDB to search the policy clauses and retrieve relevant product evidence. The LLM uses this information to select suitable policies and explain why they match the requirements.
Again, the workflow does not blindly trust the LLM. It independently verifies that every recommended policy exists in the deterministic SQL candidate set and that the recommendation has supporting evidence.
Finally, the Finalize stage combines everything into the final advisory result: the client’s archetype, benchmark, calculated financial gaps, product requirements, recommendations, assumptions, warnings, and observations.
So at a high level:

Raw Client Profile
        │
        ▼
┌──────────────────┐
│    Normalize     │
│  Standardize data│
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│     Classify     │
│   LLM Agent      │◄──── SQLite / ChromaDB
│ + deterministic  │
│    validation    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│    Calculate     │
│ Financial Gaps   │
│  Deterministic   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Requirements   │
│ Gap → Insurance  │
│    Requirement   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│    Recommend     │
│   LLM Agent      │◄──── SQLite / ChromaDB
│ + deterministic  │
│    validation    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│     Finalize     │
│   AdvisoryResult │
└──────────────────┘



## How to set up the dependencies required for the paraplanner to work?
The paraplanner relies on a local SQLite and Chroma DB (A vector store) to work. I chose this just for the POC first, for the actual project, we can host it somewhere.
In data ingestion, you will find scripts to ingest policies and archetype details. Both of these can be found on our shared Google drive. The policy summary docs are taken from
comparefirst.sg (search Great Eastern). The different client profile archetypes are from https://www.moneysens and Le.gov.sg/planning-your-finances-well/. Download the pdfs from google drive
and run the scripts in data ingestion to ingest them. For this to work, u need to put yr Gemini API Key in .env. (there is a free tier). .env to place in project root

## Limitations
Currently, only have Endowment and Life insurance policies. Need ILPs and medical/health insurance documents.
I plan to expand classify llm responsibilities to also calculate the financial gap. currently a bit of redundancy between determinstic code decision gates vs LLM. Ideally the LLM
should have access to all the deterministic code tools and make the decision. Probs change the classifier agent to a financial needs agent and add a manager agent to format
clean output back to the user.
Have not tested with other archetypes yet, only the example json found
