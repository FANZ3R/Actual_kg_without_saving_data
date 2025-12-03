import streamlit as st
import re
import logging
import time
import requests
import json
import os

# Import FastKGQuerier
from src.query.fast_querier import FastKGQuerier

# ⚡ UPDATED: Friend's Neo4j connection
NEO4J_URI = "bolt://192.168.9.175:7687"  # ← Friend's Neo4j
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "vipani@123"  # ← Friend's password

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3:latest"

# OpenRouter configuration
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "openai/gpt-oss-120b"  # Free model!

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
                "HTTP-Referer": "https://github.com/your-app",  # Optional
                "X-Title": "KG RAG Chatbot"  # Optional
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
                    "num_ctx": 4096,  # Increased for more context
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

# ========== NEO4J CONNECTION ==========
@st.cache_resource
def init_fast_querier():
    try:
        querier = FastKGQuerier(
            neo4j_uri=NEO4J_URI,
            neo4j_user=NEO4J_USERNAME,
            neo4j_password=NEO4J_PASSWORD
        )
        querier.semantic_query("test", top_k=1)
        logger.info("✅ Neo4j connected")
        return querier
    except Exception as e:
        logger.error(f"❌ Neo4j failed: {e}")
        return None

querier = init_fast_querier()

# ========== SEARCH FUNCTIONS ==========
def extract_keywords_simple(question: str):
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

def multi_strategy_search(question: str, top_k: int = 20):  # Increased to 20 for more context
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
    if len(all_results) < 8:  # Increased threshold
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

def query_knowledge_graph(question: str, top_k: int = 20):  # Increased to 20
    if not querier:
        return []
    
    start_time = time.time()
    
    try:
        results = multi_strategy_search(question, top_k=top_k)
        formatted_facts = []
        
        for item in results:
            entity = item.get('entity', '') or 'Unknown'
            label = item.get('label', '') or 'Entity'
            score = item.get('score', 0)
            neighbors = item.get('neighbors', []) or []
            
            try:
                score_val = float(score) if score is not None else 0.0
                formatted_facts.append(f"[{label}] {entity} (score: {score_val:.2f})")
            except:
                formatted_facts.append(f"[{label}] {entity}")
            
            # Show more neighbors for more context
            for neighbor in neighbors[:3]:  # Increased from 2 to 3
                rel_type = neighbor.get('type', '') or 'RELATED'
                neighbor_text = neighbor.get('text', '') or 'Unknown'
                formatted_facts.append(f"  → {entity} --[{rel_type}]--> {neighbor_text}")
        
        elapsed = time.time() - start_time
        logger.info(f"Query: {elapsed:.2f}s - {len(formatted_facts)} facts")
        return formatted_facts
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return []

def create_simple_answer(context: list) -> str:
    """Create a simple formatted answer from facts"""
    if not context:
        return "No information found."
    
    answer = "Based on the knowledge graph:\n\n"
    
    for fact in context[:15]:  # Show more facts
        if "[" in fact and "]" in fact:
            answer += f"• {fact}\n"
        elif "→" in fact:
            answer += f"  {fact}\n"
    
    return answer

def generate_answer_detailed(question: str, context: list, client, provider: str):
    """
    Generate DETAILED answer with comprehensive explanations
    """
    if not context:
        return " No relevant information found in the knowledge graph.\n\nTry:\n• Different keywords\n• Rephrasing your question\n• Asking about: supplier risk, procurement, processes"
    
    # Use MORE context for detailed answers
    context_text = "\n".join(context[:20])  # Increased from 12 to 20
    
    # Enhanced prompt for detailed responses
    prompt = f"""You are an expert knowledge assistant with access to a comprehensive knowledge graph about business processes, procurement, risk management, and organizational data.

USER QUESTION: {question}

KNOWLEDGE GRAPH FACTS:
{context_text}

INSTRUCTIONS:
1. Provide a COMPREHENSIVE and DETAILED answer to the user's question
2. Use ALL relevant facts from the knowledge graph
3. Explain concepts thoroughly with examples where applicable
4. Describe relationships and connections between entities
5. Include specific details, processes, and context
6. Structure your answer with clear paragraphs for readability
7. If multiple aspects are covered, address each one
8. Aim for 4-6 sentences minimum for a complete explanation
9. Only use information from the knowledge graph above
10. If no relevant information is found, inform the user without fabricating details
11. Dont use external knowledge not in the facts
Provide a detailed, informative response:"""
    
    # Show what we're doing
    with st.spinner(f" Generating detailed AI answer using {provider}..."):
        answer = client.generate(prompt, temperature=0.3, max_tokens=800)  # Increased tokens for detailed answers
    
    # If LLM worked, return it
    if answer and len(answer) > 20:
        logger.info(f" {provider} generated {len(answer)} chars")
        return answer
    
    # If LLM failed, return formatted facts
    logger.warning(f" {provider} failed, using fallback")
    fallback = create_simple_answer(context)
    fallback += f"\n\n Note: AI summary unavailable from {provider}. Showing raw facts instead."
    if provider == "Ollama":
        fallback += "\nTo fix: Run `ollama run llama3:latest` in a separate terminal."
    else:
        fallback += "\nCheck your OpenRouter API key or try Ollama."
    return fallback

# ========== STREAMLIT UI ==========
st.set_page_config(
    page_title="🤖 Enhanced KG RAG", 
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Enhanced KG RAG with Detailed Responses")
st.caption("⚡ Supports OpenRouter (Free GPT-OSS) & Ollama | Comprehensive Answers")

# Sidebar
with st.sidebar:
    st.header(" Configuration")
    
    # Provider selection
    provider = st.radio(
        "Select AI Provider:",
        ["OpenRouter (Free GPT-OSS)", "Ollama (Local)"],
        help="OpenRouter uses free cloud models. Ollama uses your local models."
    )
    
    st.divider()
    
    # OpenRouter API Key input
    if "OpenRouter" in provider:
        st.subheader(" OpenRouter API Key")
        
        # Check if key exists in session state or environment
        default_key = st.session_state.get('openrouter_api_key', os.getenv('OPENROUTER_API_KEY', ''))
        
        api_key = st.text_input(
            "API Key:",
            value=default_key,
            type="password",
            help="Get your free API key from https://openrouter.ai/keys"
        )
        
        if api_key:
            st.session_state.openrouter_api_key = api_key
            st.success(" API Key Set")
            
            # Model selection for OpenRouter
            openrouter_models = [
                "openai/gpt-oss-120b",  # Free!
                "meta-llama/llama-3-8b-instruct:free",  # Free!
                "mistralai/mistral-7b-instruct:free",  # Free!
            ]
            
            selected_model = st.selectbox(
                "Model:",
                openrouter_models,
                help="All these models are FREE to use!"
            )
            
            st.session_state.openrouter_model = selected_model
            
        else:
            st.warning(" Please enter your OpenRouter API key")
            st.markdown("Get a free key: [openrouter.ai/keys](https://openrouter.ai/keys)")
    
    else:
        st.subheader(" Ollama Settings")
        st.text(f"Model: {OLLAMA_MODEL}")
        st.info("Make sure Ollama is running:\n`ollama run llama3:latest`")
    
    st.divider()
    
    # Status section
    st.header(" Status")
    
    if querier:
        st.success(" Neo4j Connected")
    else:
        st.error(" Neo4j Offline")
    
    # Test provider
    if "OpenRouter" in provider:
        if api_key:
            st.success(" OpenRouter Ready")
        else:
            st.error(" API Key Missing")
    else:
        try:
            test_response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
            if test_response.status_code == 200:
                st.success(" Ollama Connected")
            else:
                st.warning(" Ollama Issue")
        except:
            st.error(" Ollama Offline")
    
    st.divider()
    
    # Response settings
    st.header(" Response Settings")
    
    detail_level = st.select_slider(
        "Detail Level:",
        options=["Concise", "Standard", "Detailed", "Comprehensive"],
        value="Detailed",
        help="How detailed should the AI responses be?"
    )
    
    st.session_state.detail_level = detail_level
    
    show_facts = st.checkbox("Show Retrieved Facts", value=True)
    st.session_state.show_facts = show_facts
    
    st.divider()
    
    st.subheader(" Tips")
    st.text("• Detailed mode gives 4-6+ sentences")
    st.text("• GPT-OSS-120B is completely FREE")
    st.text("• Retrieves up to 20 facts per query")
    st.text("• All responses include context")
    
    # Test button
    if st.button(" Test AI Provider"):
        with st.spinner("Testing..."):
            if "OpenRouter" in provider:
                if api_key:
                    test_client = OpenRouterClient(api_key, st.session_state.get('openrouter_model', OPENROUTER_MODEL))
                    test_answer = test_client.generate("Say 'OK'", max_tokens=10)
                    if test_answer:
                        st.success(f" Working! Got: {test_answer[:50]}")
                    else:
                        st.error(" Failed - check API key")
                else:
                    st.error(" Please enter API key first")
            else:
                test_client = SimpleOllamaClient()
                test_answer = test_client.generate("Say 'OK'", max_tokens=10)
                if test_answer:
                    st.success(f" Working! Got: {test_answer[:50]}")
                else:
                    st.error(" Failed - check Ollama")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "detail_level" not in st.session_state:
    st.session_state.detail_level = "Detailed"

if "show_facts" not in st.session_state:
    st.session_state.show_facts = True

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "metadata" in message and st.session_state.show_facts:
            with st.expander(" View retrieved facts"):
                for fact in message["metadata"].get("facts", []):
                    st.text(fact)

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        # Search phase
        search_start = time.time()
        with st.spinner(" Searching knowledge graph..."):
            context = query_knowledge_graph(prompt, top_k=20)
        search_time = time.time() - search_start
        
        # Show search results
        if context:
            st.success(f"⚡ Found {len(context)} facts in {search_time:.2f}s")
        else:
            st.warning(" No facts found")
        
        # Initialize client based on provider
        if "OpenRouter" in provider:
            api_key = st.session_state.get('openrouter_api_key', '')
            if not api_key:
                st.error(" Please enter your OpenRouter API key in the sidebar!")
                st.stop()
            
            model = st.session_state.get('openrouter_model', OPENROUTER_MODEL)
            client = OpenRouterClient(api_key, model)
            provider_name = f"OpenRouter ({model})"
        else:
            client = SimpleOllamaClient()
            provider_name = "Ollama"
        
        # Generate answer
        answer_start = time.time()
        response = generate_answer_detailed(prompt, context, client, provider_name)
        answer_time = time.time() - answer_start
        
        # Display answer
        st.markdown(response)
        
        # Show timing
        total_time = search_time + answer_time
        
        # Colored timing based on speed
        if total_time < 2:
            st.success(f" Total: {total_time:.2f}s (Search: {search_time:.2f}s, Answer: {answer_time:.2f}s)")
        elif total_time < 5:
            st.info(f" Total: {total_time:.2f}s (Search: {search_time:.2f}s, Answer: {answer_time:.2f}s)")
        else:
            st.warning(f" Total: {total_time:.2f}s (Search: {search_time:.2f}s, Answer: {answer_time:.2f}s)")
        
        # Show context in expander
        if context and st.session_state.show_facts:
            with st.expander(f" View {len(context)} facts from knowledge graph"):
                for fact in context:
                    st.text(fact)
    
    # Save assistant response with metadata
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response,
        "metadata": {
            "facts": context,
            "search_time": search_time,
            "answer_time": answer_time,
            "provider": provider_name
        }
    })

# Show status at bottom
st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    if querier:
        st.text(" Neo4j: Connected")
    else:
        st.text(" Neo4j: Offline")

with col2:
    if "OpenRouter" in provider:
        if st.session_state.get('openrouter_api_key'):
            st.text(" OpenRouter: Ready")
        else:
            st.text(" OpenRouter: No API Key")
    else:
        try:
            test = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=1)
            st.text(" Ollama: Connected")
        except:
            st.text(" Ollama: Offline")

with col3:
    st.text(f"Mode: {st.session_state.detail_level}")