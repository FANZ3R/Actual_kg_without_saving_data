"""
Improved Chatbot with Better LLM Synthesis
Fixes the "Information is not available" inconsistency issue
"""

import streamlit as st
import os
from typing import List, Dict, Any
import logging
import time
import traceback
from dotenv import load_dotenv

load_dotenv()

# Import dependencies
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

try:
    from qdrant_client import QdrantClient
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Configuration
CONFIG = {
    "vector": {
        "qdrant_url": os.getenv('QDRANT_URL', 'http://localhost:6333'),
        "collection_name": os.getenv('DEFAULT_COLLECTION_NAME', 'chatbot_embeddings'),
        "embedding_model": os.getenv('LOCAL_EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
    },
    "kg": {
        "neo4j_uri": os.getenv('KG_NEO4J_URI', 'bolt://localhost:7692'),
        "username": os.getenv('KG_NEO4J_USERNAME', 'neo4j'),
        "password": os.getenv('KG_NEO4J_PASSWORD', '12345678')
    },
    "llm": {
        "api_key": os.getenv('OPENROUTER_API_KEY'),
        "model": os.getenv('LLM_MODEL', "meta-llama/llama-3-70b-instruct"),
        "temperature": float(os.getenv('LLM_TEMPERATURE', '0.5')),  # Increased from 0.3
        "max_tokens": int(os.getenv('LLM_MAX_TOKENS', '1500'))  # Increased from 1000
    }
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GENERIC_ENTITIES = {
    'that', 'this', 'these', 'those', 'it', 'they', 'them', 'which', 'who', 
    'what', 'where', 'when', 'why', 'how', 'a', 'an', 'the', 'some', 'any',
    'all', 'each', 'every', 'both', 'many', 'few', 'several', 'one', 'two'
}

class LocalEmbeddingModel:
    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = os.getenv('LOCAL_EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def embed_text(self, text):
        if isinstance(text, str):
            return self.model.encode(text).tolist()
        else:
            return self.model.encode(text).tolist()

class VectorSearcher:
    def __init__(self):
        self.connected = False
        self.error_message = ""
        
        if not QDRANT_AVAILABLE or not SENTENCE_TRANSFORMERS_AVAILABLE:
            self.error_message = "Missing dependencies"
            return
        
        try:
            self.qdrant_client = QdrantClient(url=CONFIG["vector"]["qdrant_url"])
            self.collection_name = CONFIG["vector"]["collection_name"]
            
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                self.error_message = f"Collection '{self.collection_name}' not found"
                return
            
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            self.total_points = collection_info.points_count
            
            self.embedding_model = LocalEmbeddingModel(CONFIG["vector"]["embedding_model"])
            self.connected = True
            logger.info(f"✅ Vector: {self.collection_name} with {self.total_points:,} points")
            
        except Exception as e:
            self.error_message = f"Vector connection failed: {str(e)}"
            logger.error(traceback.format_exc())
    
    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        if not self.connected:
            return []
        
        try:
            query_embedding = self.embedding_model.embed_text(query)
            
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=0.3,
                with_payload=True
            )
            
            results = []
            for result in search_results:
                results.append({
                    'content': result.payload.get('text', ''),
                    'header': result.payload.get('header', 'No header'),
                    'field_name': result.payload.get('field_name', 'unknown'),
                    'score': float(result.score),
                    'source_type': 'vector_search'
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

class OptimizedKnowledgeGraphSearcher:
    def __init__(self):
        self.connected = False
        self.error_message = ""
        
        if not NEO4J_AVAILABLE:
            self.error_message = "Neo4j driver not available"
            return
        
        try:
            self.driver = GraphDatabase.driver(
                CONFIG["kg"]["neo4j_uri"], 
                auth=(CONFIG["kg"]["username"], CONFIG["kg"]["password"])
            )
            
            with self.driver.session() as session:
                result = session.run("MATCH (n:Entity) RETURN count(n) as count LIMIT 1")
                count = result.single()["count"]
                
                result = session.run("SHOW INDEXES")
                indexes = list(result)
                self.has_fulltext = any('fulltext' in str(idx.get('name', '')).lower() for idx in indexes)
            
            self.connected = True
            logger.info(f"✅ KG: {count:,} entities, fulltext: {self.has_fulltext}")
            
        except Exception as e:
            self.error_message = f"Neo4j connection failed: {str(e)}"
            logger.error(traceback.format_exc())
    
    def __del__(self):
        if hasattr(self, 'driver') and self.driver:
            try:
                self.driver.close()
            except:
                pass
    
    def is_meaningful_entity(self, text: str) -> bool:
        if not text or len(text) < 3:
            return False
        text_lower = text.lower().strip()
        if text_lower in GENERIC_ENTITIES:
            return False
        words = text_lower.split()
        if len(words) == 2 and words[0] in {'a', 'an', 'the'}:
            return False
        meaningful_words = [w for w in words if w not in GENERIC_ENTITIES and len(w) > 2]
        return len(meaningful_words) >= 1
    
    def search(self, query: str, limit: int = 15) -> List[Dict[str, Any]]:
        if not self.connected:
            return []
        
        start_time = time.time()
        
        try:
            with self.driver.session() as session:
                if self.has_fulltext:
                    results = self._fulltext_search(session, query, limit * 3)
                else:
                    results = self._contains_search(session, query, limit * 3)
                
                filtered_results = []
                for result in results:
                    source_text = result.get('source_entity', '')
                    target_text = result.get('target_entity', '')
                    
                    if self.is_meaningful_entity(source_text) and self.is_meaningful_entity(target_text):
                        filtered_results.append(result)
                    
                    if len(filtered_results) >= limit:
                        break
                
                elapsed = time.time() - start_time
                logger.info(f"KG: {len(results)} raw → {len(filtered_results)} filtered in {elapsed:.2f}s")
                
                return filtered_results[:limit]
                
        except Exception as e:
            logger.error(f"KG search failed: {e}")
            return []
    
    def _fulltext_search(self, session, query: str, limit: int) -> List[Dict[str, Any]]:
        cypher_query = """
        CALL db.index.fulltext.queryNodes('entity_fulltext_idx', $query)
        YIELD node as entity, score
        WHERE score > 0.5
        
        MATCH (entity)-[r:RELATED]->(target:Entity)
        WHERE target.text IS NOT NULL 
          AND entity.text IS NOT NULL
          AND r.confidence >= 0.6
        
        RETURN DISTINCT
            entity.text AS source_entity,
            entity.lemma AS source_lemma,
            r.type AS relationship_type,
            r.confidence AS confidence,
            target.text AS target_entity,
            target.lemma AS target_lemma,
            score AS relevance_score
        ORDER BY score DESC, r.confidence DESC
        LIMIT $limit
        """
        
        try:
            result = session.run(cypher_query, query=query, limit=limit)
            return self._format_results(result, 'fulltext')
        except Exception as e:
            logger.warning(f"Fulltext failed, fallback: {e}")
            return self._contains_search(session, query, limit)
    
    def _contains_search(self, session, query: str, limit: int) -> List[Dict[str, Any]]:
        keywords = [word for word in query.lower().split() if len(word) > 3]
        if not keywords:
            keywords = [query.lower()]
        main_keyword = max(keywords, key=len)
        
        cypher_query = """
        MATCH (source:Entity)-[r:RELATED]->(target:Entity)
        WHERE (toLower(source.text) CONTAINS toLower($keyword)
               OR toLower(source.lemma) CONTAINS toLower($keyword)
               OR toLower(target.text) CONTAINS toLower($keyword)
               OR toLower(target.lemma) CONTAINS toLower($keyword))
          AND source.text IS NOT NULL
          AND target.text IS NOT NULL
          AND r.confidence >= 0.6
        
        RETURN DISTINCT
            source.text AS source_entity,
            source.lemma AS source_lemma,
            r.type AS relationship_type,
            r.confidence AS confidence,
            target.text AS target_entity,
            target.lemma AS target_lemma
        ORDER BY r.confidence DESC
        LIMIT $limit
        """
        
        result = session.run(cypher_query, keyword=main_keyword, limit=limit)
        return self._format_results(result, 'contains')
    
    def _format_results(self, result, strategy: str) -> List[Dict[str, Any]]:
        formatted = []
        for record in result:
            try:
                formatted.append({
                    'source_entity': record.get('source_entity', ''),
                    'target_entity': record.get('target_entity', ''),
                    'relationship_type': record.get('relationship_type', 'RELATED'),
                    'confidence': float(record.get('confidence', 0)),
                    'relevance_score': float(record.get('relevance_score', 0)) if 'relevance_score' in record else 0,
                    'relationship_text': f"{record.get('source_entity')} --[{record.get('relationship_type')}]--> {record.get('target_entity')}",
                    'source_type': 'knowledge_graph',
                    'strategy': strategy
                })
            except Exception as e:
                continue
        return formatted
    
    def get_statistics(self) -> Dict[str, Any]:
        if not self.connected:
            return {}
        
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (n:Entity)
                    OPTIONAL MATCH (n)-[r:RELATED]-()
                    RETURN count(DISTINCT n) as total_entities,
                           count(r) as total_relationships
                """)
                stats = result.single()
                
                result = session.run("""
                    MATCH ()-[r:RELATED]->()
                    WHERE r.type IS NOT NULL
                    RETURN r.type as type, count(*) as count
                    ORDER BY count DESC
                    LIMIT 10
                """)
                top_rel_types = [(record["type"], record["count"]) for record in result]
                
                return {
                    "total_entities": stats["total_entities"],
                    "total_relationships": stats["total_relationships"],
                    "top_relationship_types": top_rel_types,
                    "has_fulltext_index": self.has_fulltext
                }
        except Exception as e:
            return {}

def generate_unified_response(query: str, vector_results: List[Dict], kg_results: List[Dict]) -> str:
    """
    IMPROVED: Better LLM prompt that forces synthesis even with partial information
    """
    
    if not OPENAI_AVAILABLE or not CONFIG["llm"]["api_key"]:
        # Fallback without LLM
        response_parts = [f"Query: {query}\n"]
        
        if vector_results:
            response_parts.append("Based on the documents:")
            for r in vector_results[:3]:
                response_parts.append(f"• {r['content'][:150]}...")
        
        if kg_results:
            response_parts.append("\nRelated concepts:")
            for r in kg_results[:5]:
                response_parts.append(f"• {r['relationship_text']}")
        
        return "\n".join(response_parts) if response_parts else "No results found."
    
    # Format context - IMPROVED to show more content
    context_parts = []
    
    if vector_results:
        context_parts.append("=== DOCUMENT EXCERPTS ===")
        for i, r in enumerate(vector_results[:8], 1):  # Increased from 5 to 8
            context_parts.append(f"\n[Document {i}] (Relevance: {r['score']:.3f})")
            context_parts.append(f"Field: {r['field_name']}")
            context_parts.append(f"Header: {r['header']}")
            context_parts.append(f"Content: {r['content'][:600]}...")  # Increased from 400 to 600
            context_parts.append("")
    
    if kg_results:
        context_parts.append("=== KNOWLEDGE GRAPH RELATIONSHIPS ===")
        for i, r in enumerate(kg_results[:12], 1):  # Increased from 8 to 12
            context_parts.append(f"[{i}] {r['source_entity']} --[{r['relationship_type']}]--> {r['target_entity']} (Confidence: {r['confidence']:.2f})")
        context_parts.append("")
    
    context_text = "\n".join(context_parts)
    
    # IMPROVED PROMPT - More explicit instructions
    prompt = f"""You are a procurement and business process expert with access to document content and knowledge graph relationships.

USER QUESTION: {query}

AVAILABLE INFORMATION:
{context_text}

CRITICAL INSTRUCTIONS:
1. You MUST provide a substantive answer using the information above
2. NEVER say "information is not available" if any context is provided
3. If the documents contain relevant content, extract and explain it
4. If the knowledge graph shows relationships, describe the connections
5. Synthesize BOTH sources into a coherent, helpful response
6. If information is incomplete, provide what IS available and acknowledge limitations
7. Structure your answer clearly with proper context
8. Be specific and actionable - give concrete information, not vague statements

RESPONSE GUIDELINES:
- Start directly answering the question
- Use "The documents indicate..." or "Based on the knowledge graph..." to cite sources
- If asked for steps/procedures, extract or infer them from the content
- If asked for definitions, provide clear explanations using available content
- Connect related concepts shown in the knowledge graph
- Be comprehensive but concise

Provide your answer now:"""
    
    try:
        client = OpenAI(
            api_key=CONFIG["llm"]["api_key"],
            base_url="https://openrouter.ai/api/v1"
        )
        
        response = client.chat.completions.create(
            model=CONFIG["llm"]["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=CONFIG["llm"]["temperature"],
            max_tokens=CONFIG["llm"]["max_tokens"]
        )
        
        answer = response.choices[0].message.content
        
        # DEBUG: Log if LLM says "not available" despite having results
        if ("not available" in answer.lower() or "information is not available" in answer.lower()):
            logger.warning(f"LLM claimed no info despite {len(vector_results)} vector + {len(kg_results)} KG results!")
            logger.warning(f"Query was: {query}")
            logger.warning(f"First vector result: {vector_results[0]['content'][:200] if vector_results else 'None'}")
        
        return answer
        
    except Exception as e:
        logger.error(f"LLM failed: {e}")
        return f"Found {len(vector_results)} documents and {len(kg_results)} relationships, but couldn't generate response: {e}"

# ========== STREAMLIT UI ==========

st.set_page_config(
    page_title="🚀 Procurement Assistant", 
    page_icon="🤖",
    layout="wide"
)

st.title("🚀 Procurement Knowledge Assistant")
st.caption("💡 Improved LLM Synthesis + 40M+ Relationships")

@st.cache_resource
def init_searchers():
    vector_searcher = VectorSearcher()
    kg_searcher = OptimizedKnowledgeGraphSearcher()
    return vector_searcher, kg_searcher

vector_searcher, kg_searcher = init_searchers()

# Sidebar
with st.sidebar:
    st.header("📊 System Status")
    
    if vector_searcher.connected:
        st.success("✅ Vector Search")
        if hasattr(vector_searcher, 'total_points'):
            st.metric("Documents", f"{vector_searcher.total_points:,}")
    else:
        st.error("❌ Vector Offline")
    
    if kg_searcher.connected:
        st.success("✅ Knowledge Graph")
        stats = kg_searcher.get_statistics()
        
        if stats:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Entities", f"{stats.get('total_entities', 0):,}")
            with col2:
                st.metric("Relationships", f"{stats.get('total_relationships', 0):,}")
            
            if stats.get('has_fulltext_index'):
                st.success("⚡ Fulltext Index")
            
            if stats.get('top_relationship_types'):
                with st.expander("Top Relationship Types"):
                    for rel_type, count in stats['top_relationship_types'][:5]:
                        st.text(f"• {rel_type}: {count:,}")
    else:
        st.error("❌ KG Offline")
    
    if CONFIG["llm"]["api_key"]:
        st.success("✅ LLM Configured")
        st.caption(f"Model: {CONFIG['llm']['model']}")
        st.caption(f"Temp: {CONFIG['llm']['temperature']}")
    else:
        st.warning("⚠️ LLM Not Configured")
    
    st.divider()
    
    st.subheader("⚙️ Settings")
    vector_limit = st.slider("Vector Results", 1, 15, 8)
    kg_limit = st.slider("KG Results", 5, 25, 15)
    
    st.divider()
    st.subheader("💡 Try asking:")
    st.text("• What are the steps of procurement?")
    st.text("• How to assess supplier risk?")
    st.text("• What is agile procurement?")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Ask about procurement..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.write(prompt)
    
    with st.chat_message("assistant"):
        if not vector_searcher.connected and not kg_searcher.connected:
            response = "Systems offline. Check configuration."
            st.error(response)
        else:
            with st.spinner("🔍 Searching..."):
                search_start = time.time()
                
                vector_results = []
                kg_results = []
                
                if vector_searcher.connected:
                    try:
                        vector_results = vector_searcher.search(prompt, limit=vector_limit)
                    except Exception as e:
                        st.warning(f"Vector: {e}")
                
                if kg_searcher.connected:
                    try:
                        kg_results = kg_searcher.search(prompt, limit=kg_limit)
                    except Exception as e:
                        st.warning(f"KG: {e}")
                
                search_time = time.time() - search_start
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📄 Vector", len(vector_results))
            with col2:
                st.metric("🕸️ KG", len(kg_results))
            with col3:
                st.metric("⏱️ Search", f"{search_time:.2f}s")
            
            with st.spinner("💭 Generating answer..."):
                response_start = time.time()
                response = generate_unified_response(prompt, vector_results, kg_results)
                response_time = time.time() - response_start
                st.write(response)
            
            st.caption(f"ℹ️ Total: {search_time + response_time:.2f}s (Search: {search_time:.2f}s, Answer: {response_time:.2f}s)")
            
            if vector_results or kg_results:
                with st.expander(f"🔍 View detailed sources ({len(vector_results)} vector + {len(kg_results)} KG)"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📄 Vector Results")
                        if vector_results:
                            for i, r in enumerate(vector_results, 1):
                                st.markdown(f"**{i}.** Score: `{r['score']:.3f}`")
                                st.markdown(f"*{r['header'][:100]}*")
                                st.caption(r['content'][:200] + "...")
                                st.divider()
                        else:
                            st.info("No vector results")
                    
                    with col2:
                        st.subheader("🕸️ KG Results")
                        if kg_results:
                            for i, r in enumerate(kg_results, 1):
                                st.markdown(f"**{i}.** `{r['confidence']:.2f}`")
                                st.markdown(f"*{r['relationship_text']}*")
                                st.divider()
                        else:
                            st.info("No KG results")
    
    st.session_state.messages.append({"role": "assistant", "content": response})

st.markdown("---")
st.caption("🤖 Enhanced RAG | Improved LLM Synthesis | 40M+ Relationships")