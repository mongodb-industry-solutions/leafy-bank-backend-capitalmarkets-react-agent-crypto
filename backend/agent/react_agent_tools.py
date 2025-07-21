import os
import logging
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_tavily import TavilySearch
from agent.vogayeai.vogaye_ai_embeddings import VogayeAIEmbeddings
from agent.db.mdb import MongoDBConnector

# Initialize dotenv to load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize search tool
tavily_search_tool = TavilySearch(max_results=3)

# Initialize embeddings
embedding_model_id = os.getenv("EMBEDDINGS_MODEL_ID", "voyage-finance-2")
ve = VogayeAIEmbeddings(api_key=os.getenv("VOYAGE_API_KEY"))

# Getting environment variables for MongoDB collections
CRYPTO_ANALYSIS_COLLECTION_NAME = os.getenv("REPORTS_COLLECTION_CRYPTO_ANALYSIS", "reports_crypto_analysis")
CRYPTO_NEWS_COLLECTION_NAME = os.getenv("REPORTS_COLLECTION_CRYPTO_NEWS", "reports_crypto_news")
CRYPTO_SM_COLLECTION_NAME = os.getenv("REPORTS_COLLECTION_CRYPTO_SM", "reports_crypto_sm")
PORTFOLIO_ALLOCATION_COLLECTION_NAME = os.getenv("CRYPTO_PORTFOLIO_ALLOCATION_COLLECTION", "crypto_portfolio_allocation")
PORTFOLIO_PERFORMANCE_COLLECTION_NAME = os.getenv("PORTFOLIO_PERFORMANCE_COLLECTION", "portfolio_performance")

# Initialize MongoDB connector
mongodb_connector = MongoDBConnector()

# Get the collections for crypto reports
crypto_analysis_collection = mongodb_connector.get_collection(collection_name=CRYPTO_ANALYSIS_COLLECTION_NAME)
crypto_news_collection = mongodb_connector.get_collection(collection_name=CRYPTO_NEWS_COLLECTION_NAME)
crypto_sm_collection = mongodb_connector.get_collection(collection_name=CRYPTO_SM_COLLECTION_NAME)

# Get portfolio allocation and performance collections
portfolio_allocation_collection = mongodb_connector.get_collection(collection_name=PORTFOLIO_ALLOCATION_COLLECTION_NAME)
portfolio_performance_collection = mongodb_connector.get_collection(PORTFOLIO_PERFORMANCE_COLLECTION_NAME)

# Getting environment variables for vector index names
REPORT_CRYPTO_ANALYSIS_VECTOR_INDEX_NAME = os.getenv("REPORT_CRYPTO_ANALYSIS_VECTOR_INDEX_NAME")
REPORT_CRYPTO_NEWS_VECTOR_INDEX_NAME = os.getenv("REPORT_CRYPTO_NEWS_VECTOR_INDEX_NAME")
REPORT_CRYPTO_SM_VECTOR_INDEX_NAME = os.getenv("REPORT_CRYPTO_SM_VECTOR_INDEX_NAME")

# Getting environment variables for vector field names
REPORT_VECTOR_FIELD = os.getenv("REPORT_VECTOR_FIELD", "report_embedding")


@tool
def crypto_analysis_reports_vector_search_tool(query: str, k: int = 1):
    """
    Perform a vector similarity search on crypto analysis reports for the CURRENT CRYPTO PORTFOLIO.

    IMPORTANT: This tool should typically be used AFTER get_portfolio_allocation_tool
    to provide context-aware technical analysis for the user's actual holdings.

    COMMON USAGE PATTERNS:
    - After get_portfolio_allocation_tool for reallocation advice
    - After get_portfolio_allocation_tool for technical analysis questions
    - Part of comprehensive market analysis (with news and social tools)

    Use this tool when you need:
    - Crypto trends and momentum analysis for portfolio assets
    - Technical indicators for crypto assets (RSI, moving averages, etc.)
    - Market conditions to inform reallocation decisions
    - Crypto-specific diagnostics for portfolio holdings
    - Market volatility analysis for cryptocurrency assets

    Args:
        query (str): The search query related to crypto portfolio assets, trends, momentum, etc.
        k (int, optional): The number of top results to return. Defaults to 1.

    Returns:
        dict: Contains relevant sections from the most recent crypto analysis report for the current portfolio.
    """
    try:
        logger.info(f"Searching crypto portfolio analysis for: {query}")
        
        # Get the most recent document for context information
        most_recent = crypto_analysis_collection.find_one(
            {}, 
            sort=[("timestamp", -1)]
        )
        
        if not most_recent:
            return {"results": "No crypto analysis reports found for the current portfolio."}
        
        # Extract the date of the most recent report
        report_date = most_recent.get("date_string", "Unknown date")
        
        # Get portfolio assets list for context
        portfolio_assets = []
        try:
            portfolio_allocation = portfolio_allocation_collection.find({})
            portfolio_assets = [asset.get("symbol", "Unknown") for asset in portfolio_allocation]
        except Exception as e:
            logger.warning(f"Could not retrieve portfolio assets: {str(e)}")

        # Generate query embedding
        query_embedding = ve.get_embeddings(model_id=embedding_model_id, text=query)
        
        # Perform vector search across all documents
        pipeline = [
            {
                "$vectorSearch": {
                    "index": f"{REPORT_CRYPTO_ANALYSIS_VECTOR_INDEX_NAME}",
                    "path": f"{REPORT_VECTOR_FIELD}",
                    "queryVector": query_embedding,
                    "numCandidates": 50,
                    "limit": k
                }
            }
        ]
        
        results = list(crypto_analysis_collection.aggregate(pipeline))
        
        if not results:
            return {"results": "No matching crypto analysis found for your query."}
        
        # Format results for the agent
        formatted_results = []
        for result in results:
            report = result.get("report", {})
            formatted_result = {
                "date": result.get("date_string", report_date),
                "crypto_trends": report.get("crypto_trends", []),
                "momentum_indicators": report.get("crypto_momentum_indicators", []),
                "overall_diagnosis": report.get("overall_diagnosis", ""),
                "portfolio_assets": portfolio_assets
            }
            formatted_results.append(formatted_result)
        
        return {
            "results": formatted_results,
            "portfolio_context": f"Analysis for crypto portfolio containing: {', '.join(portfolio_assets)}",
            "report_date": report_date
        }
        
    except Exception as e:
        logger.error(f"Error in crypto_analysis_reports_vector_search_tool: {str(e)}")
        return {"results": f"Error searching crypto analysis reports: {str(e)}"}


@tool
def crypto_news_reports_vector_search_tool(query: str, k: int = 1):
    """
    Perform a vector similarity search on crypto news reports for the CURRENT CRYPTO PORTFOLIO.

    IMPORTANT: This tool should typically be used AFTER get_portfolio_allocation_tool.
    For sentiment questions, use this WITH crypto_social_media_reports_vector_search_tool.

    COMMON USAGE PATTERNS:
    - After get_portfolio_allocation_tool for sentiment analysis
    - Combined with crypto_social_media_reports_vector_search_tool for complete sentiment picture
    - Part of comprehensive market analysis

    Use this tool when you need:
    - Recent crypto news affecting portfolio assets
    - News sentiment analysis for crypto portfolio holdings  
    - Media coverage impact on portfolio assets
    - News-driven market sentiment for crypto assets

    Args:
        query (str): The search query related to crypto portfolio assets.
        k (int, optional): The number of top results to return. Defaults to 1.

    Returns:
        dict: Contains relevant news summaries from the most recent crypto news reports for the current portfolio.
    """
    try:
        logger.info(f"Searching crypto news reports for: {query}")
        
        # Get the most recent document for context information
        most_recent = crypto_news_collection.find_one(
            {}, 
            sort=[("timestamp", -1)]
        )
        
        if not most_recent:
            return {"results": "No crypto news reports found for the current portfolio."}
        
        # Extract the date of the most recent report
        report_date = most_recent.get("date_string", "Unknown date")
        
        # Get portfolio assets list for context
        portfolio_assets = []
        try:
            portfolio_allocation = portfolio_allocation_collection.find({})
            portfolio_assets = [asset.get("symbol", "Unknown") for asset in portfolio_allocation]
        except Exception as e:
            logger.warning(f"Could not retrieve portfolio assets: {str(e)}")

        # Generate query embedding
        query_embedding = ve.get_embeddings(model_id=embedding_model_id, text=query)
        
        # Perform vector search across all documents
        pipeline = [
            {
                "$vectorSearch": {
                    "index": f"{REPORT_CRYPTO_NEWS_VECTOR_INDEX_NAME}",
                    "path": f"{REPORT_VECTOR_FIELD}",
                    "queryVector": query_embedding,
                    "numCandidates": 50,
                    "limit": k
                }
            }
        ]
        
        results = list(crypto_news_collection.aggregate(pipeline))
        
        if not results:
            return {"results": "No matching crypto news found for your query."}
        
        # Format results for the agent
        formatted_results = []
        for result in results:
            report = result.get("report", {})
            formatted_result = {
                "date": result.get("date_string", report_date),
                "asset_news": report.get("asset_news", []),
                "news_sentiments": report.get("asset_news_sentiments", []),
                "overall_news_diagnosis": report.get("overall_news_diagnosis", ""),
                "portfolio_assets": portfolio_assets
            }
            formatted_results.append(formatted_result)
        
        return {
            "results": formatted_results,
            "portfolio_context": f"News analysis for crypto portfolio containing: {', '.join(portfolio_assets)}",
            "report_date": report_date
        }
        
    except Exception as e:
        logger.error(f"Error in crypto_news_reports_vector_search_tool: {str(e)}")
        return {"results": f"Error searching crypto news reports: {str(e)}"}


@tool
def crypto_social_media_reports_vector_search_tool(query: str, k: int = 1):
    """
    Perform a vector similarity search on crypto social media sentiment reports for the CURRENT CRYPTO PORTFOLIO.

    IMPORTANT: This tool should typically be used AFTER get_portfolio_allocation_tool.
    For complete sentiment analysis, use this WITH crypto_news_reports_vector_search_tool.

    COMMON USAGE PATTERNS:
    - After get_portfolio_allocation_tool for sentiment analysis
    - Combined with crypto_news_reports_vector_search_tool for complete sentiment picture
    - Essential for crypto markets where social sentiment drives prices

    Use this tool when you need:
    - Social media sentiment analysis for crypto portfolio assets
    - Community perception and discussions about portfolio holdings
    - Reddit, Twitter, and other social platform sentiment
    - Social media-driven market sentiment for crypto assets
    - Community-based insights affecting portfolio assets

    Args:
        query (str): The search query related to crypto social media sentiment.
        k (int, optional): The number of top results to return. Defaults to 1.

    Returns:
        dict: Contains relevant social media sentiment analysis from the most recent reports
              for the current crypto portfolio.
    """
    try:
        logger.info(f"Searching crypto social media sentiment for: {query}")
        
        # Get the most recent document for context information
        most_recent = crypto_sm_collection.find_one(
            {}, 
            sort=[("timestamp", -1)]
        )
        
        if not most_recent:
            return {"results": "No crypto social media sentiment reports found for the current portfolio."}
        
        # Extract the date of the most recent report
        report_date = most_recent.get("date_string", "Unknown date")
        
        # Get portfolio assets list for context
        portfolio_assets = []
        try:
            portfolio_allocation = portfolio_allocation_collection.find({})
            portfolio_assets = [asset.get("symbol", "Unknown") for asset in portfolio_allocation]
        except Exception as e:
            logger.warning(f"Could not retrieve portfolio assets: {str(e)}")

        # Generate query embedding
        query_embedding = ve.get_embeddings(model_id=embedding_model_id, text=query)
        
        # Perform vector search across all documents
        pipeline = [
            {
                "$vectorSearch": {
                    "index": f"{REPORT_CRYPTO_SM_VECTOR_INDEX_NAME}",
                    "path": f"{REPORT_VECTOR_FIELD}",
                    "queryVector": query_embedding,
                    "numCandidates": 50,
                    "limit": k
                }
            }
        ]
        
        results = list(crypto_sm_collection.aggregate(pipeline))
        
        if not results:
            return {"results": "No matching crypto social media sentiment found for your query."}
        
        # Format results for the agent
        formatted_results = []
        for result in results:
            report = result.get("report", {})
            formatted_result = {
                "date": result.get("date_string", report_date),
                "asset_subreddits": report.get("asset_subreddits", []),
                "social_media_sentiments": report.get("asset_sm_sentiments", []),
                "overall_social_diagnosis": report.get("overall_news_diagnosis", ""),
                "portfolio_assets": portfolio_assets
            }
            formatted_results.append(formatted_result)
        
        return {
            "results": formatted_results,
            "portfolio_context": f"Social media sentiment for crypto portfolio containing: {', '.join(portfolio_assets)}",
            "report_date": report_date
        }
        
    except Exception as e:
        logger.error(f"Error in crypto_social_media_reports_vector_search_tool: {str(e)}")
        return {"results": f"Error searching crypto social media reports: {str(e)}"}


@tool
def get_portfolio_allocation_tool(query: str) -> dict:
    """Get the most recent crypto portfolio allocation.

    CRITICAL: This tool should be called FIRST for ANY portfolio-related question.
    The user cannot get meaningful advice about "their" portfolio without knowing what they own.

    MANDATORY USAGE PATTERNS:
    - Reallocation questions → Use this FIRST, then crypto_analysis_reports_vector_search_tool
    - Sentiment questions → Use this FIRST, then crypto_news + crypto_social_media tools  
    - Performance questions → Use this FIRST, then get_portfolio_ytd_return_tool
    - ANY "my portfolio" question → Use this FIRST

    Use this tool when you need:
    - Crypto portfolio allocation for the current portfolio
    - Digital asset distribution information  
    - Current cryptocurrency investment breakdown
    - Asset types (Cryptocurrency vs Stablecoin)
    - To confirm what assets the user owns before providing advice

    Args:
        query (str): The search query related to crypto portfolio allocation.

    Returns:
        dict: Crypto portfolio allocation showing assets, descriptions, and percentages.
    """
    try:
        logger.info(f"Getting crypto portfolio allocation for: {query}")
        
        # Get all portfolio allocation documents
        portfolio_allocation = list(portfolio_allocation_collection.find({}))
        
        if not portfolio_allocation:
            return {"results": "No crypto portfolio allocation found."}
        
        # Format the allocation data
        formatted_allocation = []
        for asset in portfolio_allocation:
            formatted_asset = {
                "symbol": asset.get("symbol", "Unknown"),
                "description": asset.get("description", "Unknown"),
                "allocation_percentage": asset.get("allocation_percentage", "0%"),
                "allocation_number": asset.get("allocation_number", 0),
                "asset_type": asset.get("asset_type", "Unknown"),
                "binance_symbol": asset.get("binance_symbol", "Unknown")
            }
            formatted_allocation.append(formatted_asset)
        
        return {
            "results": formatted_allocation,
            "total_assets": len(formatted_allocation),
            "portfolio_type": "Cryptocurrency Portfolio"
        }
        
    except Exception as e:
        logger.error(f"Error in get_portfolio_allocation_tool: {str(e)}")
        return {"results": f"Error retrieving crypto portfolio allocation: {str(e)}"}


@tool
def get_portfolio_ytd_return_tool(query: str) -> str:
    """
    Get the Year-to-Date (YTD) rate of return for the crypto portfolio.

    IMPORTANT: This tool provides the YTD performance of the current crypto portfolio.

    Use this tool when you need:
    - Year-to-date return of the crypto portfolio
    - YTD crypto portfolio performance 
    - How the crypto portfolio has performed since the beginning of this year

    Args:
        query (str): The search query related to crypto portfolio YTD return.

    Returns:
        str: Crypto portfolio YTD return percentage and information.
    """
    try:
        logger.info(f"Getting crypto portfolio YTD return for: {query}")
        
        # Get the most recent performance document
        most_recent = portfolio_performance_collection.find_one(
            {}, 
            sort=[("_id", -1)]
        )
        
        if not most_recent:
            return "No crypto portfolio performance data found."
        
        # Extract the YTD return
        ytd_return = most_recent.get("percentage_of_cumulative_return", 0)
        
        # Convert to percentage format
        ytd_return_percentage = round(ytd_return * 100, 2)
        
        return f"The current Year-to-Date (YTD) return for your crypto portfolio is {ytd_return_percentage}%."
        
    except Exception as e:
        logger.error(f"Error in get_portfolio_ytd_return_tool: {str(e)}")
        return f"Error retrieving crypto portfolio YTD return: {str(e)}"
