"""
Unified Knowledge Graph + Vector Database Chatbot - Core Python Version
No Streamlit dependencies - Pure Python for API integration

This is the "engine" that powers the chatbot.
It can be wrapped with FastAPI, Flask, or used directly in Python scripts.
"""

import re
import logging
import time
import requests
from typing import List, Dict, Any, Optional
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ========================================
# CORE CONFIGURATION
# ========================================
class ChatbotConfig:
    """Configuration for the chatbot system"""
    
    def __init__(self):
        # Neo4j Knowledge Graph
        self.neo4j_uri = os.getenv('NEO4J_URI', "bolt://192.168.9.175:7687")
        self.neo4j_username = os.getenv('NEO4J_USERNAME', "neo4j")
        self.neo4j_password = os.getenv('NEO4J_PASSWORD', "vipani@123")
        
        # Qdrant Vector Database
        self.qdrant_url = os.getenv('QDRANT_URL', 'http://localhost:6333')
        self.collection_name = os.getenv('COLLECTION_NAME', 'chatbot')
        self.embedding_model = os.getenv('EMBEDDING_MODEL', 'BAAI/bge-base-en-v1.5')
        
        # OpenRouter API
        self.openrouter_base_url = "https://openrouter.ai/api/v1"
        self.openrouter_model = os.getenv('OPENROUTER_MODEL', "openai/gpt-oss-120b")
        self.openrouter_api_key = os.getenv('OPENROUTER_API_KEY', '')
        
        # Search settings
        self.vector_limit = 12
        self.kg_limit = 20
        self.vector_score_threshold = 0.25
        
        # LLM settings
        self.temperature = 0.15
        self.max_tokens = 1200


# ========================================
# OPENROUTER CLIENT
# ========================================
class OpenRouterClient:
    """Client for OpenRouter API"""
    
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"
    
    def generate(self, prompt: str, temperature: float = 0.15, max_tokens: int = 1200) -> Optional[str]:
        """Generate response using OpenRouter API"""
        try:
            url = f"{self.base_url}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/unified-kg-vector-rag",
                "X-Title": "Unified KG+Vector RAG Chatbot"
            }
            
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            logger.info(f"Calling OpenRouter ({self.model})...")
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            answer = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            logger.info(f"Got response: {len(answer)} characters")
            return answer
            
        except Exception as e:
            logger.error(f"OpenRouter error: {e}")
            return None


# ========================================
# LOCAL EMBEDDING MODEL
# ========================================
class LocalEmbeddingModel:
    """Local embedding model using sentence-transformers"""
    
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Dimension: {self.embedding_dim}")
    
    def embed_text(self, text: str) -> List[float]:
        """Create embedding for text"""
        embedding = self.model.encode(text)
        return embedding.tolist()


# ========================================
# VECTOR SEARCHER
# ========================================
class VectorSearcher:
    """Vector database searcher using Qdrant"""
    
    def __init__(self, config: ChatbotConfig):
        from qdrant_client import QdrantClient
        
        self.config = config
        self.qdrant_client = QdrantClient(url=config.qdrant_url)
        self.collection_name = config.collection_name
        
        # Verify collection exists
        collections = self.qdrant_client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if self.collection_name not in collection_names:
            raise ValueError(f"Collection '{self.collection_name}' not found")
        
        # Get collection info
        collection_info = self.qdrant_client.get_collection(self.collection_name)
        self.total_points = collection_info.points_count
        
        # Initialize embedding model
        self.embedding_model = LocalEmbeddingModel(config.embedding_model)
        
        logger.info(f"✅ Vector system initialized: {self.total_points:,} vectors")
    
    def search(self, query: str, limit: int = 12, score_threshold: float = 0.25) -> List[Dict[str, Any]]:
        """Search vector database"""
        try:
            # Create query embedding
            query_embedding = self.embedding_model.embed_text(query)
            
            # Search Qdrant
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True
            )
            
            # Format results
            results = []
            for result in search_results:
                payload = result.payload
                
                results.append({
                    'content': payload.get('text', '')[:400],  # Display truncated
                    'full_content': payload.get('text', ''),   # Full for LLM
                    'header': payload.get('header', 'No header')[:100],
                    'field_name': f"{payload.get('file_type', 'unknown')}: {payload.get('content_type', 'general')}",
                    'score': float(result.score),
                    'source_type': 'vector_db',
                    'source_file': payload.get('source_file', 'unknown')
                })
            
            logger.info(f"Vector search: {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []


# ========================================
# KNOWLEDGE GRAPH SEARCHER
# ========================================
class KnowledgeGraphSearcher:
    """Knowledge graph searcher using FastKGQuerier"""
    
    def __init__(self, config: ChatbotConfig):
        from src.query.fast_querier import FastKGQuerier
        
        self.config = config
        self.querier = FastKGQuerier(
            neo4j_uri=config.neo4j_uri,
            neo4j_user=config.neo4j_username,
            neo4j_password=config.neo4j_password
        )
        
        # Test connection
        self.querier.semantic_query("test", top_k=1)
        logger.info("✅ Knowledge Graph connected")
    
    def extract_keywords(self, question: str) -> str:
        """Extract keywords from question"""
        stopwords = {
            "what", "is", "a", "an", "the", "who", "where", "when", "why", "how",
            "in", "of", "and", "to", "for", "with", "by", "from", "up", "about",
            "tell", "me", "show", "find", "get", "give", "explain", "describe"
        }
        
        words = re.findall(r'\w+', question.lower())
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        return " ".join(keywords)
    
    def search(self, question: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Search knowledge graph with multi-strategy approach"""
        try:
            all_results = []
            seen_entities = set()
            
            # Strategy 1: Direct question search
            results1 = self.querier.semantic_query(
                query_text=question,
                top_k=top_k,
                include_neighbors=True,
                use_cache=True
            )
            
            for item in results1:
                entity = item.get('entity', '')
                if entity and entity not in seen_entities:
                    all_results.append(item)
                    seen_entities.add(entity)
            
            # Strategy 2: Keyword search
            keywords = self.extract_keywords(question)
            if keywords and keywords != question.lower():
                results2 = self.querier.semantic_query(
                    query_text=keywords,
                    top_k=top_k,
                    include_neighbors=True,
                    use_cache=True
                )
                
                for item in results2:
                    entity = item.get('entity', '')
                    if entity and entity not in seen_entities:
                        all_results.append(item)
                        seen_entities.add(entity)
            
            # Format results
            formatted_facts = []
            for item in all_results:
                entity = item.get('entity', '') or 'Unknown'
                label = item.get('label', '') or 'Entity'
                score = float(item.get('score', 0))
                neighbors = item.get('neighbors', []) or []
                
                # Add main entity
                formatted_facts.append({
                    'text': f"[{label}] {entity}",
                    'score': score,
                    'source_type': 'knowledge_graph'
                })
                
                # Add relationships
                for neighbor in neighbors[:5]:
                    rel_type = neighbor.get('type', '') or 'RELATED'
                    neighbor_text = neighbor.get('text', '') or 'Unknown'
                    formatted_facts.append({
                        'text': f"  → {entity} --[{rel_type}]--> {neighbor_text}",
                        'score': score * 0.9,
                        'source_type': 'knowledge_graph'
                    })
            
            logger.info(f"KG search: {len(formatted_facts)} results")
            return formatted_facts[:top_k]
            
        except Exception as e:
            logger.error(f"KG search failed: {e}")
            return []


# ========================================
# UNIFIED CHATBOT - MAIN CLASS
# ========================================
class UnifiedChatbot:
    """
    Main chatbot class that combines Knowledge Graph and Vector Database
    
    This is the core engine - no UI dependencies
    Can be used directly in Python or wrapped with an API
    """
    
    def __init__(self, config: Optional[ChatbotConfig] = None):
        """Initialize the chatbot with configuration"""
        self.config = config or ChatbotConfig()
        
        # Initialize components
        self.vector_searcher = None
        self.kg_searcher = None
        self.llm_client = None
        
        # Initialize systems
        self._initialize_systems()
    
    def _initialize_systems(self):
        """Initialize all systems (Vector DB, KG, LLM)"""
        logger.info("Initializing Unified Chatbot...")
        
        # Initialize Vector Searcher
        try:
            self.vector_searcher = VectorSearcher(self.config)
        except Exception as e:
            logger.warning(f"Vector system initialization failed: {e}")
        
        # Initialize Knowledge Graph Searcher
        try:
            self.kg_searcher = KnowledgeGraphSearcher(self.config)
        except Exception as e:
            logger.warning(f"KG system initialization failed: {e}")
        
        # Initialize LLM Client
        if self.config.openrouter_api_key:
            self.llm_client = OpenRouterClient(
                self.config.openrouter_api_key,
                self.config.openrouter_model
            )
            logger.info("✅ LLM client initialized")
        else:
            logger.warning("⚠️ No OpenRouter API key provided")
        
        # Check if at least one system is available
        if not self.vector_searcher and not self.kg_searcher:
            raise RuntimeError("No search systems available (neither Vector nor KG)")
        
        logger.info("✅ Chatbot initialization complete")
    
    def search_databases(self, question: str) -> Dict[str, Any]:
        """
        Search both databases and return results
        
        Args:
            question: User's question
            
        Returns:
            Dict with vector_results, kg_results, and timing info
        """
        start_time = time.time()
        
        vector_results = []
        kg_results = []
        
        # Search Vector Database
        if self.vector_searcher:
            try:
                vec_start = time.time()
                vector_results = self.vector_searcher.search(
                    question,
                    limit=self.config.vector_limit,
                    score_threshold=self.config.vector_score_threshold
                )
                vector_time = time.time() - vec_start
            except Exception as e:
                logger.error(f"Vector search error: {e}")
                vector_time = 0
        else:
            vector_time = 0
        
        # Search Knowledge Graph
        if self.kg_searcher:
            try:
                kg_start = time.time()
                kg_results = self.kg_searcher.search(
                    question,
                    top_k=self.config.kg_limit
                )
                kg_time = time.time() - kg_start
            except Exception as e:
                logger.error(f"KG search error: {e}")
                kg_time = 0
        else:
            kg_time = 0
        
        total_time = time.time() - start_time
        
        return {
            'vector_results': vector_results,
            'kg_results': kg_results,
            'timing': {
                'vector_time': vector_time,
                'kg_time': kg_time,
                'total_search_time': total_time
            }
        }
    
    def generate_answer(self, question: str, vector_results: List[Dict], 
                       kg_results: List[Dict]) -> Optional[str]:
        """
        Generate answer using LLM based on search results
        
        Args:
            question: User's question
            vector_results: Results from vector search
            kg_results: Results from knowledge graph search
            
        Returns:
            Generated answer string
        """
        if not self.llm_client:
            return "LLM client not initialized. Please provide OpenRouter API key."
        
        # Prepare context
        context_parts = []
        
        # Add vector results
        if vector_results:
            context_parts.append("=== INFORMATION FROM VECTOR DATABASE ===")
            for i, result in enumerate(vector_results[:12], 1):
                full_text = result.get('full_content', result.get('content', ''))
                context_parts.append(f"{i}. [{result['field_name']}] {full_text[:600]}")
        
        # Add KG results
        if kg_results:
            context_parts.append("\n=== INFORMATION FROM KNOWLEDGE GRAPH ===")
            for i, result in enumerate(kg_results[:20], 1):
                context_parts.append(f"{i}. {result['text']}")
        
        context = "\n".join(context_parts)
        
        # Check if we have any results
        if not vector_results and not kg_results:
            return "I couldn't find any relevant information in the database to answer your question."
        
        # Create prompt
        prompt = f"""You are a knowledgeable assistant that provides detailed, accurate answers based EXCLUSIVELY on information from internal databases.

CRITICAL INSTRUCTIONS:
1. You MUST use ONLY the information provided below from the databases
2. NEVER add external knowledge or make assumptions beyond the provided data
3. DO NOT include source attribution tags like "(Knowledge Graph)", "(Vector Database)", etc.
4. If information is available, provide a comprehensive answer (5-6 sentences minimum)
5. ONLY say "information is not available" if there are truly NO relevant results
6. Synthesize information from all sources naturally
7. Be specific and include concrete details

Question: {question}

Available Information from Databases:
{context}

Provide a detailed, comprehensive answer using ALL relevant information above. Write naturally without source attribution tags:"""
        
        # Generate answer
        try:
            answer = self.llm_client.generate(
                prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            if not answer:
                return "I apologize, but I'm having trouble generating a response. Please try again."
            
            # Clean up any source tags
            answer = re.sub(r'\((?:Knowledge Graph|Vector Database|Vector DB|KG|Sources?:)[^)]*\)', '', answer)
            answer = re.sub(r'\[(?:Knowledge Graph|Vector Database|Vector DB|KG|Sources?:)[^\]]*\]', '', answer)
            answer = re.sub(r'\s+', ' ', answer).strip()
            
            return answer
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return f"Error generating answer: {str(e)}"
    
    def ask(self, question: str) -> Dict[str, Any]:
        """
        Main method: Ask a question and get a complete response
        
        This is the main entry point for the API
        
        Args:
            question: User's question
            
        Returns:
            Dict containing:
                - answer: The generated answer
                - vector_results: List of vector search results
                - kg_results: List of KG search results
                - timing: Dict with timing information
        """
        logger.info(f"Processing question: {question}")
        
        # Search databases
        search_results = self.search_databases(question)
        
        # Generate answer
        answer_start = time.time()
        answer = self.generate_answer(
            question,
            search_results['vector_results'],
            search_results['kg_results']
        )
        answer_time = time.time() - answer_start
        
        # Prepare response
        response = {
            'question': question,
            'answer': answer,
            'vector_results': search_results['vector_results'],
            'kg_results': search_results['kg_results'],
            'metadata': {
                'vector_count': len(search_results['vector_results']),
                'kg_count': len(search_results['kg_results']),
                'search_time': search_results['timing']['total_search_time'],
                'answer_time': answer_time,
                'total_time': search_results['timing']['total_search_time'] + answer_time
            }
        }
        
        logger.info(f"Response generated in {response['metadata']['total_time']:.2f}s")
        return response


# ========================================
# EXAMPLE USAGE
# ========================================
if __name__ == "__main__":
    """
    Example of how to use the Unified Chatbot directly in Python
    """
    
    # Create configuration
    config = ChatbotConfig()
    config.openrouter_api_key = "YOUR_API_KEY_HERE"  # Set your API key
    
    # Initialize chatbot
    print("Initializing chatbot...")
    chatbot = UnifiedChatbot(config)
    
    # Ask a question
    question = "What are the procurement best practices?"
    print(f"\nQuestion: {question}")
    
    response = chatbot.ask(question)
    
    # Print answer
    print(f"\nAnswer: {response['answer']}")
    
    # Print metadata
    print(f"\nMetadata:")
    print(f"  - Vector results: {response['metadata']['vector_count']}")
    print(f"  - KG results: {response['metadata']['kg_count']}")
    print(f"  - Search time: {response['metadata']['search_time']:.2f}s")
    print(f"  - Answer time: {response['metadata']['answer_time']:.2f}s")
    print(f"  - Total time: {response['metadata']['total_time']:.2f}s")