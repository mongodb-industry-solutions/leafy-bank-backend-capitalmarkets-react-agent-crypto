import logging
import os
from dotenv import load_dotenv
from agent.db.mdb import MongoDBConnector

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class AgentProfiles(MongoDBConnector):
    def __init__(self, collection_name: str = None, uri: str = None, database_name: str = None, appname: str = None):
        """
        AgentProfiles class to retrieve agent profiles from MongoDB.

        Args:
            collection_name (str, optional): Collection name. Default is None and will be retrieved from the config: AGENT_PROFILES_COLLECTION.
            uri (str, optional): MongoDB URI. Default parent class value.
            database_name (str, optional): Database name. Default parent class value.
            appname (str, optional): Application name. Default parent class value.
        """
        super().__init__(uri, database_name, appname)
        self.collection_name = collection_name or os.getenv("AGENT_PROFILES_COLLECTION", "agent_profiles")
        self.collection = self.get_collection(self.collection_name)
        logger.info("AgentProfiles initialized")

    def get_agent_profile(self, agent_id: str) -> dict:
        """
        Retrieve the agent profile for the given agent ID.

        Args:
            agent_id (str): Agent ID to retrieve the profile for.

        Returns:
            dict: Agent profile for the given agent ID.
        """
        try:
            # Retrieve the agent profile from MongoDB
            profile = self.collection.find_one({"agent_id": agent_id})
            if profile:
                # Remove the MongoDB ObjectId from the profile
                del profile["_id"]
                # Log the successful retrieval of the profile
                logger.info(f"Agent profile found for agent ID: {agent_id}")
                return profile
            else:
                logger.warning(f"No profile found for agent ID: {agent_id}")
                return None
        except Exception as e:
            logger.error(f"Error retrieving agent profile: {e}")
            return None
            
    def generate_system_prompt(self, agent_id: str) -> str:
        """
        Retrieve the system prompt for the given agent ID.

        Args:
            agent_id (str): Agent ID to retrieve the system prompt for.

        Returns:
            str: System prompt for the given agent ID.
        """
        # Retrieve the agent profile using the class's method
        profile = self.get_agent_profile(agent_id)
        
        if not profile:
            return "You are a helpful assistant. Answer questions to the best of your ability."
        
        # Construct the system prompt
        system_prompt = f"""
            # {profile.get('profile', 'Assistant')}

            You are {profile.get('role', 'a crypto portfolio assistant')} with the purpose of {profile.get('motive', 'helping users optimize their cryptocurrency investments')}.

            ## Core Operating Principles

            1. **Portfolio First**: ALWAYS start with get_portfolio_allocation_tool for any portfolio-related question
            2. **Multi-Tool Analysis**: Use multiple relevant tools to provide comprehensive insights
            3. **Tool Progression**: Each suggested next step must use a DIFFERENT tool than those already used
            4. **Single Next Step**: End EVERY response with exactly ONE follow-up question (YES/NO format)

            ## Available Tools & Usage Patterns

            - **get_portfolio_allocation_tool**: Current holdings (use FIRST for portfolio questions)
            - **crypto_analysis_reports_vector_search_tool**: Technical analysis, trends, momentum
            - **crypto_news_reports_vector_search_tool**: News sentiment analysis
            - **crypto_social_media_reports_vector_search_tool**: Social media sentiment
            - **get_portfolio_ytd_return_tool**: Year-to-date performance
            - **tavily_search_tool**: General crypto research

            Select tools based on the question's focus - sentiment needs news+social, technical needs analysis, etc.

            ## YES/NO Response Recognition

            Before processing any message, check if it's a YES/NO response to your previous suggestion:
            - Affirmative: yes, yeah, sure, ok, go ahead, do it, yes please
            - Negative: no, nope, nah, not now, no thanks, skip
            
            If YES → Execute the suggested action with NEW tools and insights
            If NO → Acknowledge and pivot to a different type of assistance

            ## How to Provide Analysis

            1. **Be Decisive**: Convert technical indicators (RSI, trends) into clear recommendations
            2. **Use Specific Percentages**: "Reduce BTC to 40-50%" not "reduce BTC allocation"
            3. **Interpret Data**: Oversold = buying opportunity, bearish = reduce exposure
            4. **Build Progressively**: Each YES response should deepen analysis with different tools

            ## Response Structure

            Every response must include:
            1. **Analysis**: Address the user's question using appropriate tools
            2. **Recommendations**: Specific, actionable advice with percentages
            3. **Next Step**: ONE question that would use a DIFFERENT tool
            
            Format: "Would you like me to [action that uses new tool]? YES/NO"
            
            Tool Progression Examples:
            - After allocation+analysis → "check your YTD performance?"
            - After allocation+news → "analyze social media sentiment?"
            - After using all crypto tools → "research regulatory updates?"
            
            {profile.get('instructions', '')}
            {profile.get('rules', '')}
            {profile.get('goals', '')}
        """
        
        return system_prompt.strip()