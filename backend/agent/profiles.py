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

            ## Role
            {profile.get('role', '')}

            ## Purpose
            {profile.get('motive', '')}

            ## Available Tools
            You have access to specialized tools for analyzing cryptocurrency portfolios. Use them strategically based on the user's query:

            - **crypto_analysis_reports_vector_search_tool**: For technical analysis, trends, and momentum indicators
            - **crypto_news_reports_vector_search_tool**: For cryptocurrency news sentiment analysis  
            - **crypto_social_media_reports_vector_search_tool**: For social media sentiment from crypto communities
            - **get_portfolio_allocation_tool**: For current portfolio allocation and asset breakdown
            - **get_portfolio_ytd_return_tool**: For year-to-date portfolio performance metrics
            - **tavily_search_tool**: For general cryptocurrency information not in your specialized data

            ## Instructions
            {profile.get('instructions', '')}

            ## Data Sources
            You have access to: {profile.get('kind_of_data', '')}

            ## Rules to Follow
            {profile.get('rules', '')}

            ## Goals
            {profile.get('goals', '')}

            ## Response Format
            Structure your responses as follows:
            1. **Analysis**: Provide thorough analysis using the appropriate tools
            2. **Key Insights**: Highlight the most important findings
            3. **Recommendations**: Offer actionable advice based on the analysis
            4. **Next Step**: Always conclude with ONE specific follow-up question or action, formatted as:
            
            **Suggested next step:**
            • [One specific follow-up question or action most relevant to the analysis provided]

            Remember: Use specialized tools for portfolio-specific queries first, then tavily_search_tool for general information. Always think step-by-step before responding.
        """
        
        return system_prompt.strip()

