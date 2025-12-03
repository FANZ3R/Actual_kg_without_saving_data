"""
Unified KG + Vector Chatbot with OpenRouter API
Combines:
- Knowledge Graph search using FastKGQuerier (from chatbot_openrouter_api.py)
- Vector Database search using Qdrant (from improved_unified_chatbot.py)
- OpenRouter/Ollama LLM support
- Answers generated ONLY from your database (KG + Vector)
"""

import streamlit as st
import re
import logging
import time
import requests
import json
import os
from typing import List, Dict, Any
import traceback

# Import FastKGQuerier for Knowledge Graph
try:
    from src.query.fast_querier import FastKGQuerier
    FAST_KG_AVAILABLE = True
except ImportError as e:
    st.error(f"FastKGQuerier not available: {e}")
    FAST_KG_AVAILABLE = False

# Import Qdrant for Vector Search
try:
    from qdrant_client import QdrantClient
    QDRANT_AVAILABLE = True
except ImportError as e:
    st.warning(f"Qdrant not available: {e}")
    QDRANT_AVAILABLE = False

# Import SentenceTransformer for embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    st.warning(f"SentenceTransformers not available: {e}")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# ========== CONFIGURATION ==========
# Neo4j Knowledge Graph Config
NEO4J_URI = os.getenv('NEO4J_URI', "bolt://localhost:7692")
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME', "neo4j")
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', "12345678")

# Qdrant Vector DB Config
QDRANT_URL = os.getenv('QDRANT_URL', 'http://localhost:6333')
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'chatbot_embeddings')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')

# Ollama Config
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', "http://localhost:11434")
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', "llama3:latest")

# OpenRouter Config
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = os.getenv('OPENROUTER_MODEL', "openai/gpt-oss-120b")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== OPENROUTER CLIENT ==========
class OpenRouterClient:
    """OpenRouter API client for free models"""
    
    def __init__(self, api_key: str, model: str = OPENROUTER_MODEL):
        self.api_key = api_key
        self.model = model
        self.base_url = OPENROUTER_BASE_URL
    
    def generate(self, prompt: str, temperature: float = 0.3, max_tokens: int = 800):
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
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            logger.info(f"Sending request to OpenRouter ({self.model})...")
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            answer = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            logger.info(f"Got response: {len(answer)} characters")
            return answer
            
        except requests.exceptions.Timeout:
            logger.error("OpenRouter timeout")
            return None
        except requests.exceptions.HTTPError as e:
            logger.error(f"OpenRouter HTTP error: {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"OpenRouter error: {e}")
            return None

# ========== OLLAMA CLIENT ==========
class SimpleOllamaClient:
    """Simple, reliable Ollama client"""
    
    def __init__(self, base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL):
        self.base_url = base_url
        self.model = model
        self.session = requests.Session()
    
    def generate(self, prompt: str, temperature: float = 0.1, max_tokens: int = 800):
        """Simple non-streaming generation"""
        try:
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "num_ctx": 4096,
                }
            }
            
            logger.info(f"Sending request to Ollama...")
            response = self.session.post(url, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            answer = result.get("response", "")
            logger.info(f"Got response: {len(answer)} characters")
            return answer
            
        except requests.exceptions.Timeout:
            logger.error("Timeout")
            return None
        except Exception as e:
            logger.error(f"Error: {e}")
            return None

# ========== LOCAL EMBEDDING MODEL ==========
class LocalEmbeddingModel:
    """Local embedding model using sentence-transformers"""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """Initialize local embedding model"""
        self.model_name = model_name
        logger.info(f"Loading local embedding model: {model_name}")
        
        try:
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def embed_text(self, text):
        """Create embeddings for text"""
        try:
            if isinstance(text, str):
                embedding = self.model.encode(text)
                return embedding.tolist()
            else:
                embeddings = self.model.encode(text)
                return embeddings.tolist()
        except Exception as e:
            logger.error(f"Embedding creation failed: {e}")
            raise

# ========== VECTOR SEARCHER ==========
class VectorSearcher:
    """Vector searcher using Qdrant"""
    
    def __init__(self):
        self.connected = False
        self.error_message = ""
        
        if not QDRANT_AVAILABLE:
            self.error_message = "Qdrant client not available"
            return
            
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            self.error_message = "SentenceTransformers not available"
            return
        
        try:
            self.qdrant_client = QdrantClient(url=QDRANT_URL)
            self.collection_name = COLLECTION_NAME
            
            # Test connection
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                self.error_message = f"Collection '{self.collection_name}' not found. Available: {collection_names}"
                return
            
            # Get collection info
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            self.total_points = collection_info.points_count
            
            # Initialize embedding model
            self.embedding_model = LocalEmbeddingModel(EMBEDDING_MODEL)
            self.connected = True
            logger.info(f"✅ Vector system connected: {self.collection_name} with {self.total_points:,} points")
            
        except Exception as e:
            self.error_message = f"Vector connection failed: {str(e)}"
            logger.error(f"Vector initialization error: {e}")
            logger.error(traceback.format_exc())
    
    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search vector database"""
        if not self.connected:
            return []
            
        try:
            # Create query embedding
            query_embedding = self.embedding_model.embed_text(query)
            
            # Search Qdrant
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=0.3,
                with_payload=True
            )
            
            # Format results
            results = []
            for result in search_results:
                results.append({
                    'content': result.payload.get('text', ''),
                    'header': result.payload.get('header', 'No header'),
                    'field_name': result.payload.get('field_name', 'unknown'),
                    'score': float(result.score),
                    'source_type': 'vector_db',
                    'record_id': result.payload.get('record_id', -1),
                    'chunk_index': result.payload.get('chunk_index', 0),
                    'total_chunks': result.payload.get('total_chunks', 1)
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

# ========== KNOWLEDGE GRAPH CONNECTION ==========
@st.cache_resource
def init_fast_querier():
    """Initialize FastKGQuerier for Knowledge Graph"""
    if not FAST_KG_AVAILABLE:
        return None
        
    try:
        querier = FastKGQuerier(
            neo4j_uri=NEO4J_URI,
            neo4j_user=NEO4J_USERNAME,
            neo4j_password=NEO4J_PASSWORD
        )
        querier.semantic_query("test", top_k=1)
        logger.info("✅ Neo4j Knowledge Graph connected")
        return querier
    except Exception as e:
        logger.error(f"❌ Neo4j failed: {e}")
        return None

# ========== KNOWLEDGE GRAPH SEARCH FUNCTIONS ==========
def extract_keywords_simple(question: str):
    """Extract keywords from question"""
    stopwords = {
        "what", "is", "a", "an", "the", "who", "where", "when", "why", "how",
        "in", "of", "and", "to", "for", "with", "by", "from", "up", "about",
        "into", "through", "during", "before", "after", "above", "below",
        "do", "does", "are", "was", "were", "been", "have", "has", "had",
        "will", "would", "could", "should", "may", "might", "can", "?", "!",
        "tell", "me", "show", "find", "get", "give", "explain", "describe",
        "define", "mean", "means", "meaning"
    }
    
    words = re.findall(r'\w+', question.lower())
    keywords = [w for w in words if w not in stopwords and len(w) > 2]
    return " ".join(keywords)

def multi_strategy_kg_search(question: str, querier, top_k: int = 15):
    """Multi-strategy search for Knowledge Graph"""
    if not querier:
        return []
    
    all_results = []
    seen_entities = set()
    
    # Direct search
    results1 = querier.semantic_query(
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
    
    # Keyword search if needed
    if len(all_results) < 8:
        keywords = extract_keywords_simple(question)
        if keywords != question.lower():
            results2 = querier.semantic_query(
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
    
    all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
    return all_results[:top_k]

def search_knowledge_graph(question: str, querier, top_k: int = 15):
    """Search Knowledge Graph and return formatted facts"""
    if not querier:
        return []
    
    try:
        results = multi_strategy_kg_search(question, querier, top_k=top_k)
        formatted_facts = []
        
        for item in results:
            entity = item.get('entity', '') or 'Unknown'
            label = item.get('label', '') or 'Entity'
            score = item.get('score', 0)
            neighbors = item.get('neighbors', []) or []
            
            try:
                score_val = float(score) if score is not None else 0.0
                formatted_facts.append({
                    'text': f"[{label}] {entity}",
                    'score': score_val,
                    'source_type': 'knowledge_graph'
                })
            except:
                formatted_facts.append({
                    'text': f"[{label}] {entity}",
                    'score': 0.0,
                    'source_type': 'knowledge_graph'
                })
            
            # Add relationship context
            for neighbor in neighbors[:3]:
                rel_type = neighbor.get('type', '') or 'RELATED'
                neighbor_text = neighbor.get('text', '') or 'Unknown'
                formatted_facts.append({
                    'text': f"  → {entity} --[{rel_type}]--> {neighbor_text}",
                    'score': score_val * 0.9,  # Slightly lower score for relationships
                    'source_type': 'knowledge_graph'
                })
        
        return formatted_facts
        
    except Exception as e:
        logger.error(f"KG search failed: {e}")
        return []

# ========== UNIFIED ANSWER GENERATION ==========
def generate_unified_answer(question: str, vector_results: List[Dict], kg_results: List[Dict], 
                           client, provider_name: str, detail_level: str = "Detailed"):
    """Generate answer using BOTH vector and KG results"""
    
    # Prepare context from both sources
    context_parts = []
    
    # Add vector search results
    if vector_results:
        context_parts.append("=== VECTOR DATABASE RESULTS ===")
        for i, result in enumerate(vector_results[:8], 1):  # Top 8 vector results
            context_parts.append(f"{i}. [{result['field_name']}] {result['content'][:300]}")
    
    # Add knowledge graph results
    if kg_results:
        context_parts.append("\n=== KNOWLEDGE GRAPH RESULTS ===")
        for i, result in enumerate(kg_results[:12], 1):  # Top 12 KG results
            context_parts.append(f"{i}. {result['text']}")
    
    context = "\n".join(context_parts)
    
    if not context_parts:
        return "I couldn't find any relevant information in the database to answer your question."
    
    # Determine response length based on detail level
    length_instructions = {
        "Concise": "Provide a brief 2-3 sentence answer.",
        "Standard": "Provide a clear 3-4 sentence answer.",
        "Detailed": "Provide a comprehensive 5-7 sentence answer with specific details.",
        "Comprehensive": "Provide an extensive 8-10 sentence answer covering all relevant aspects."
    }
    
    length_instruction = length_instructions.get(detail_level, length_instructions["Detailed"])
    
    # Create the prompt - STRICT instructions to use ONLY provided data
    prompt = f"""You are a helpful assistant that answers questions using ONLY the information provided from our internal databases (Vector Database and Knowledge Graph).

STRICT RULES:
1. Use ONLY the facts provided below from our databases
2. DO NOT use any external knowledge or make assumptions
3. If the provided data doesn't contain the answer, say "The information is not available in the database"
4. Cite which source type you're using (Vector DB or Knowledge Graph)
5. Combine information from both sources when relevant
6. {length_instruction}
7. Be specific and factual based on the retrieved data
8. Use ALL relevant facts from the knowledge graph and vector database
9. If multiple aspects are covered, address each one
10. Ensure clarity and coherence in the final answer
11. Structure your answer with clear paragraphs for readability
12. Provide a COMPREHENSIVE and DETAILED answer to the user's question
13. Check that the answer is relevant to the question asked and uses the information from the databases
Question: {question}

Retrieved Data from Our Databases:
{context}

Answer (using ONLY the above data):"""
    
    # Generate answer
    answer = client.generate(prompt, temperature=0.2, max_tokens=600)
    
    if not answer:
        return "I apologize, but I'm having trouble generating a response. Please try again."
    
    return answer

# ========== STREAMLIT UI ==========
st.set_page_config(page_title="Unified KG + Vector Chatbot", page_icon="🤖", layout="wide")

st.title("🤖 Unified Knowledge Graph + Vector Database Chatbot")
st.markdown("*Powered by OpenRouter/Ollama + Neo4j + Qdrant*")

# Initialize systems
kg_querier = init_fast_querier()
vector_searcher = VectorSearcher()

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # System Status
    st.subheader("📊 System Status")
    
    col1, col2 = st.columns(2)
    with col1:
        if kg_querier:
            st.success("✅ KG")
        else:
            st.error("❌ KG")
    
    with col2:
        if vector_searcher.connected:
            st.success("✅ Vector")
        else:
            st.error("❌ Vector")
    
    if not kg_querier and not vector_searcher.connected:
        st.error("⚠️ No systems connected!")
    
    if vector_searcher.connected:
        st.info(f"📊 {vector_searcher.total_points:,} vectors")
    
    st.divider()
    
    # Provider selection
    st.subheader("🤖 AI Provider")
    provider = st.radio(
        "Select Provider:",
        ["OpenRouter (Free)", "Ollama (Local)"],
        help="Choose between cloud (OpenRouter) or local (Ollama) AI"
    )
    
    st.divider()
    
    # OpenRouter configuration
    if "OpenRouter" in provider:
        st.subheader("🔑 OpenRouter API Key")
        
        default_key = st.session_state.get('openrouter_api_key', os.getenv('OPENROUTER_API_KEY', ''))
        
        api_key = st.text_input(
            "API Key:",
            value=default_key,
            type="password",
            help="Get your free API key from https://openrouter.ai/keys"
        )
        
        if api_key:
            st.session_state.openrouter_api_key = api_key
            st.success("✅ API Key Set")
            
            # Model selection
            openrouter_models = [
                "openai/gpt-oss-120b",
                "meta-llama/llama-3-8b-instruct:free",
                "mistralai/mistral-7b-instruct:free",
            ]
            
            selected_model = st.selectbox(
                "Model:",
                openrouter_models,
                help="All these models are FREE!"
            )
            
            st.session_state.openrouter_model = selected_model
        else:
            st.warning("⚠️ Please enter API key")
            st.markdown("[Get free key](https://openrouter.ai/keys)")
    
    else:
        st.subheader("🦙 Ollama Settings")
        st.text(f"Model: {OLLAMA_MODEL}")
        st.info("Ensure Ollama is running:\n`ollama run llama3:latest`")
    
    st.divider()
    
    # Response settings
    st.subheader("📝 Response Settings")
    
    detail_level = st.select_slider(
        "Detail Level:",
        options=["Concise", "Standard", "Detailed", "Comprehensive"],
        value="Detailed",
        help="How detailed should responses be?"
    )
    st.session_state.detail_level = detail_level
    
    # Search limits
    st.subheader("🔍 Search Limits")
    vector_limit = st.slider("Vector Results", 1, 10, 5, key="vector_limit")
    kg_limit = st.slider("KG Results", 5, 20, 15, key="kg_limit")
    
    show_sources = st.checkbox("Show Sources", value=True)
    st.session_state.show_sources = show_sources
    
    st.divider()
    
    # Test connections
    if st.button("🧪 Test Systems", use_container_width=True):
        with st.spinner("Testing..."):
            test_results = []
            
            # Test Vector
            if vector_searcher.connected:
                try:
                    start = time.time()
                    results = vector_searcher.search("test", limit=1)
                    elapsed = time.time() - start
                    test_results.append(("Vector", len(results), elapsed, "✅"))
                except Exception as e:
                    test_results.append(("Vector", 0, 0, f"❌ {str(e)[:30]}"))
            
            # Test KG
            if kg_querier:
                try:
                    start = time.time()
                    results = search_knowledge_graph("test", kg_querier, top_k=1)
                    elapsed = time.time() - start
                    test_results.append(("KG", len(results), elapsed, "✅"))
                except Exception as e:
                    test_results.append(("KG", 0, 0, f"❌ {str(e)[:30]}"))
            
            # Display results
            for system, count, time_taken, status in test_results:
                if "✅" in status:
                    st.success(f"{status} {system}: {count} results in {time_taken:.2f}s")
                else:
                    st.error(f"{system}: {status}")
    
    st.divider()
    
    st.subheader("💡 Tips")
    st.text("• Answers use ONLY your data")
    st.text("• Combines KG + Vector sources")
    st.text("• Detailed mode: 5-7 sentences")
    st.text("• Sources shown in expandable view")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "detail_level" not in st.session_state:
    st.session_state.detail_level = "Detailed"

if "show_sources" not in st.session_state:
    st.session_state.show_sources = True

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if "metadata" in message and st.session_state.show_sources:
            metadata = message["metadata"]
            with st.expander(f"🔍 View sources ({metadata.get('vector_count', 0)} vector + {metadata.get('kg_count', 0)} KG)"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📄 Vector Results**")
                    for result in metadata.get('vector_results', [])[:5]:
                        st.text(f"• [{result['field_name']}] {result['content'][:100]}...")
                
                with col2:
                    st.markdown("**🕸️ KG Results**")
                    for result in metadata.get('kg_results', [])[:5]:
                        st.text(f"• {result['text'][:100]}...")

# Chat input
if prompt := st.chat_input("Ask about your data..."):
    # Check if at least one system is connected
    if not kg_querier and not vector_searcher.connected:
        st.error("⚠️ No systems are connected. Please check your configuration.")
        st.stop()
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        # Search phase
        search_start = time.time()
        vector_results = []
        kg_results = []
        vector_time = 0
        kg_time = 0
        
        with st.spinner("🔍 Searching databases..."):
            # Vector search
            if vector_searcher.connected:
                try:
                    vec_start = time.time()
                    vector_results = vector_searcher.search(prompt, limit=vector_limit)
                    vector_time = time.time() - vec_start
                    logger.info(f"Vector search: {len(vector_results)} results in {vector_time:.2f}s")
                except Exception as e:
                    st.warning(f"Vector search failed: {e}")
            
            # Knowledge Graph search
            if kg_querier:
                try:
                    kg_start = time.time()
                    kg_results = search_knowledge_graph(prompt, kg_querier, top_k=kg_limit)
                    kg_time = time.time() - kg_start
                    logger.info(f"KG search: {len(kg_results)} results in {kg_time:.2f}s")
                except Exception as e:
                    st.warning(f"KG search failed: {e}")
        
        total_search_time = time.time() - search_start
        
        # Show search metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 Vector", len(vector_results), delta=f"{vector_time:.2f}s")
        with col2:
            st.metric("🕸️ KG", len(kg_results), delta=f"{kg_time:.2f}s")
        with col3:
            st.metric("⚡ Total Search", f"{total_search_time:.2f}s")
        
        # Initialize client
        if "OpenRouter" in provider:
            api_key = st.session_state.get('openrouter_api_key', '')
            if not api_key:
                st.error("⚠️ Please enter your OpenRouter API key in the sidebar!")
                st.stop()
            
            model = st.session_state.get('openrouter_model', OPENROUTER_MODEL)
            client = OpenRouterClient(api_key, model)
            provider_name = f"OpenRouter ({model})"
        else:
            client = SimpleOllamaClient()
            provider_name = "Ollama"
        
        # Generate answer
        answer_start = time.time()
        with st.spinner(f"💭 Generating answer with {provider_name}..."):
            response = generate_unified_answer(
                prompt, 
                vector_results, 
                kg_results, 
                client, 
                provider_name,
                st.session_state.detail_level
            )
        answer_time = time.time() - answer_start
        
        # Display answer
        st.markdown(response)
        
        # Show timing
        total_time = total_search_time + answer_time
        if total_time < 3:
            st.success(f"✅ Total: {total_time:.2f}s (Search: {total_search_time:.2f}s, Answer: {answer_time:.2f}s)")
        elif total_time < 6:
            st.info(f"ℹ️ Total: {total_time:.2f}s (Search: {total_search_time:.2f}s, Answer: {answer_time:.2f}s)")
        else:
            st.warning(f"⚠️ Total: {total_time:.2f}s (Search: {total_search_time:.2f}s, Answer: {answer_time:.2f}s)")
        
        # Show detailed sources in expander
        if (vector_results or kg_results) and st.session_state.show_sources:
            with st.expander(f"🔍 View detailed sources ({len(vector_results)} vector + {len(kg_results)} KG)"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📄 Vector Database Results")
                    if vector_results:
                        for i, result in enumerate(vector_results, 1):
                            st.markdown(f"**{i}.** Score: `{result['score']:.3f}`")
                            st.markdown(f"*Field:* {result['field_name']}")
                            st.markdown(f"*Content:* {result['content'][:200]}...")
                            st.divider()
                    else:
                        st.info("No vector results")
                
                with col2:
                    st.subheader("🕸️ Knowledge Graph Results")
                    if kg_results:
                        for i, result in enumerate(kg_results, 1):
                            st.markdown(f"**{i}.** Score: `{result['score']:.3f}`")
                            st.markdown(f"*Fact:* {result['text']}")
                            st.divider()
                    else:
                        st.info("No KG results")
    
    # Save assistant response with metadata
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "metadata": {
            "vector_results": vector_results,
            "kg_results": kg_results,
            "vector_count": len(vector_results),
            "kg_count": len(kg_results),
            "search_time": total_search_time,
            "answer_time": answer_time,
            "provider": provider_name
        }
    })

# Footer
st.divider()
col1, col2, col3, col4 = st.columns(4)

with col1:
    if kg_querier:
        st.text("🕸️ KG: Connected")
    else:
        st.text("🕸️ KG: Offline")

with col2:
    if vector_searcher.connected:
        st.text("📄 Vector: Connected")
    else:
        st.text("📄 Vector: Offline")

with col3:
    if "OpenRouter" in provider:
        if st.session_state.get('openrouter_api_key'):
            st.text("🤖 OpenRouter: Ready")
        else:
            st.text("🤖 OpenRouter: No Key")
    else:
        try:
            test = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=1)
            st.text("🤖 Ollama: Connected")
        except:
            st.text("🤖 Ollama: Offline")

with col4:
    st.text(f"📝 Mode: {st.session_state.detail_level}")

st.markdown("---")
st.caption("⚡ Unified KG + Vector RAG System | Answers generated from YOUR database only")