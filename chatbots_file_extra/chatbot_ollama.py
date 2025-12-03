import streamlit as st
import re
import logging
import time
import requests
import json

# Import FastKGQuerier
from src.query.fast_querier import FastKGQuerier

# ========== CONFIG ==========
NEO4J_URI = "bolt://localhost:7692"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "12345678"

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3:latest"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== SIMPLE OLLAMA CLIENT ==========
class SimpleOllamaClient:
    """Simple, reliable Ollama client"""
    
    def __init__(self, base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL):
        self.base_url = base_url
        self.model = model
        self.session = requests.Session()
    
    def generate(self, prompt: str, temperature: float = 0.1, max_tokens: int = 300):
        """Simple non-streaming generation that always works"""
        try:
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "num_ctx": 2048,
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

@st.cache_resource
def init_ollama_client():
    return SimpleOllamaClient()

ollama_client = init_ollama_client()

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

def multi_strategy_search(question: str, top_k: int = 15):
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
    if len(all_results) < 5:
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

def query_knowledge_graph(question: str, top_k: int = 15):
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
            
            for neighbor in neighbors[:2]:
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
    
    for fact in context[:10]:
        if "[" in fact and "]" in fact:
            # Entity line
            answer += f"• {fact}\n"
        elif "→" in fact:
            # Relationship line
            answer += f"  {fact}\n"
    
    return answer

def generate_answer(question: str, context: list):
    """Generate answer with automatic fallback"""
    if not context:
        return "❌ No relevant information found in the knowledge graph.\n\nTry:\n• Different keywords\n• Rephrasing your question\n• Asking about: supplier risk, procurement, processes"
    
    # Try LLM first
    context_text = "\n".join(context[:12])
    
    prompt = f"""Answer this question using the facts below.

Question: {question}

Facts:
{context_text}

Provide a clear answer in 2-3 sentences:"""
    
    # Show what we're doing
    with st.spinner("🤖 Generating AI answer..."):
        answer = ollama_client.generate(prompt, temperature=0.1, max_tokens=300)
    
    # If LLM worked, return it
    if answer and len(answer) > 10:
        logger.info(f"✅ LLM generated {len(answer)} chars")
        return answer
    
    # If LLM failed, return formatted facts
    logger.warning("⚠️ LLM failed, using fallback")
    fallback = create_simple_answer(context)
    fallback += "\n\n💡 Note: AI summary unavailable. Showing raw facts instead."
    fallback += "\nTo fix: Run `ollama run llama3:latest` in a separate terminal."
    return fallback

# ========== STREAMLIT UI ==========
st.set_page_config(
    page_title="🔧 Simple KG RAG", 
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Simple & Reliable KG RAG")
st.caption("⚡ Always shows results - no silent failures!")

# Sidebar
with st.sidebar:
    st.header("📊 Status")
    
    if querier:
        st.success("✅ Neo4j Connected")
    else:
        st.error("❌ Neo4j Offline")
    
    # Test Ollama
    try:
        test_response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        if test_response.status_code == 200:
            st.success("✅ Ollama Connected")
        else:
            st.warning("⚠️ Ollama Issue")
    except:
        st.error("❌ Ollama Offline")
    
    st.subheader("💡 Tips")
    st.text("• Keep terminal with")
    st.code("ollama run llama3:latest")
    st.text("  running in background")
    st.text("")
    st.text("• First answer may take 5-10s")
    st.text("• Later answers: 2-3s")
    st.text("")
    st.text("• If no AI answer, shows facts")
    
    st.subheader("🔧 Model")
    st.text(f"Using: {OLLAMA_MODEL}")
    
    if st.button("🧪 Test Ollama"):
        with st.spinner("Testing..."):
            test_answer = ollama_client.generate("Say 'OK'", max_tokens=10)
            if test_answer:
                st.success(f"✅ Working! Got: {test_answer[:50]}")
            else:
                st.error("❌ Failed - check Ollama")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

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
        with st.spinner("🔍 Searching knowledge graph..."):
            context = query_knowledge_graph(prompt, top_k=15)
        search_time = time.time() - search_start
        
        # Show search results
        if context:
            st.success(f"⚡ Found {len(context)} facts in {search_time:.2f}s")
        else:
            st.warning("⚠️ No facts found")
        
        # Generate answer
        answer_start = time.time()
        response = generate_answer(prompt, context)
        answer_time = time.time() - answer_start
        
        # Display answer
        st.markdown(response)
        
        # Show timing
        total_time = search_time + answer_time
        st.info(f"⏱️ Total: {total_time:.2f}s (Search: {search_time:.2f}s, Answer: {answer_time:.2f}s)")
        
        # Show context in expander
        if context:
            with st.expander(f"📋 View {len(context)} facts from knowledge graph"):
                for fact in context:
                    st.text(fact)
    
    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})

# Show status at bottom
st.divider()
col1, col2 = st.columns(2)

with col1:
    if querier:
        st.text("✅ Neo4j: Connected")
    else:
        st.text("❌ Neo4j: Offline")

with col2:
    try:
        test = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=1)
        st.text("✅ Ollama: Connected")
    except:
        st.text("❌ Ollama: Run 'ollama serve'")