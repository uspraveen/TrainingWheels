# retriever.py - BLAZING FAST optimized with minimal connection testing

import os
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from rich.console import Console
from rich.markdown import Markdown
from rich import print as rich_print
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
import time
import shutil
import json
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
import asyncio
from functools import partial
import threading
from queue import Queue, Empty
import copy
import dotenv

dotenv.load_dotenv()  # Load environment variables from .env file

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === CONFIG ===
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Advanced retrieval settings
DEFAULT_NUM_RESULTS = 20
MAX_QUERY_TOKENS = 15000
VECTOR_INDEX_NAME = "chunkVector"
EMBEDDING_PROPERTY = "textEmbedding"

# OPTIMIZED Performance settings - Balanced for reliability and speed
MAX_WORKERS = 6  # Reduced from 8 for more stability
BATCH_SIZE = 6   # Matched to MAX_WORKERS for optimal batching  
CONNECTION_POOL_SIZE = 4  # Reduced from 6 - smaller pool, faster & more reliable initialization

# Simple in-memory cache for query results
_query_cache = {}
CACHE_ENABLED = True

def clear_query_cache():
    """Clear the query cache - useful for development"""
    global _query_cache
    _query_cache.clear()
    logger.info("Query cache cleared")

# Store credentials in environment
os.environ["NEO4J_URI"] = NEO4J_URI
os.environ["NEO4J_USER"] = NEO4J_USER
os.environ["NEO4J_PASSWORD"] = NEO4J_PASSWORD
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Get terminal width for proper text wrapping
terminal_width = shutil.get_terminal_size().columns
if terminal_width <= 0:
    terminal_width = 80

console = Console(width=terminal_width - 4)

# Typewriter effect settings
CHAR_DELAY = 0.00
WORD_DELAY = 0.00
FAST_MODE = True

# === BLAZING FAST CONNECTION POOL ===
class OptimizedConnectionPool:
    """OPTIMIZED: Ultra-fast connection pool with minimal testing and parallel initialization."""
    
    def __init__(self, pool_size: int = CONNECTION_POOL_SIZE):
        self.pool_size = pool_size
        self._neo4j_graphs = Queue(maxsize=pool_size)
        self._llm_instances = Queue(maxsize=pool_size)
        self._embedding_instances = Queue(maxsize=pool_size)
        self._lock = threading.Lock()
        self._initialized = False
        
        # Start BLAZING FAST initialization
        self._initialization_thread = threading.Thread(target=self._fast_initialize_pools)
        self._initialization_thread.daemon = True
        self._initialization_thread.start()
    
    def _fast_initialize_pools(self):
        """BLAZING FAST: Initialize pools with minimal testing and parallel creation."""
        try:
            start_time = time.time()
            logger.info(f"🚀 BLAZING FAST pool initialization starting (size {self.pool_size})...")
            
            # OPTIMIZATION: Use ThreadPoolExecutor for parallel initialization
            with ThreadPoolExecutor(max_workers=3) as executor:
                # Submit all initialization tasks in parallel
                neo4j_future = executor.submit(self._init_neo4j_pool)
                llm_future = executor.submit(self._init_llm_pool)
                embedding_future = executor.submit(self._init_embedding_pool)
                
                # Wait for all to complete with individual error handling
                neo4j_success = True
                llm_success = True
                embedding_success = True
                
                try:
                    neo4j_success = neo4j_future.result(timeout=12)  # Increased timeout
                except Exception as e:
                    logger.warning(f"Neo4j pool init failed: {e}")
                    neo4j_success = False
                
                try:
                    llm_success = llm_future.result(timeout=10)  # Increased timeout
                except Exception as e:
                    logger.warning(f"LLM pool init failed: {e}")
                    llm_success = False
                
                try:
                    embedding_success = embedding_future.result(timeout=10)  # Increased timeout
                except Exception as e:
                    logger.warning(f"Embedding pool init failed: {e}")
                    embedding_success = False
                
                if neo4j_success and llm_success and embedding_success:
                    self._initialized = True
                    init_time = time.time() - start_time
                    logger.info(f"⚡ BLAZING FAST pool initialization complete in {init_time:.2f}s!")
                else:
                    logger.warning(f"Partial pool initialization - fallbacks will be used")
                    
        except Exception as e:
            logger.warning(f"Pool initialization failed, using fallbacks: {e}")
    
    def _init_neo4j_pool(self) -> bool:
        """Initialize Neo4j pool with SINGLE connection test"""
        try:
            # OPTIMIZATION: Test only ONE connection, then create the rest
            test_graph = Neo4jGraph(
                url=NEO4J_URI,
                username=NEO4J_USER,
                password=NEO4J_PASSWORD
            )
            test_graph.query("RETURN 1 as test")  # Single test query
            self._neo4j_graphs.put(test_graph)
            
            # Create remaining connections WITHOUT testing each one
            for _ in range(self.pool_size - 1):
                graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD)
                self._neo4j_graphs.put(graph)
            
            return True
        except Exception as e:
            logger.error(f"Neo4j pool initialization failed: {e}")
            return False
    
    def _init_llm_pool(self) -> bool:
        """Initialize LLM pool - all instances are identical, no testing needed"""
        try:
            # OPTIMIZATION: LLM instances don't need testing, create all at once
            for _ in range(self.pool_size):
                llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0)
                self._llm_instances.put(llm)
            return True
        except Exception as e:
            logger.error(f"LLM pool initialization failed: {e}")
            return False
    
    def _init_embedding_pool(self) -> bool:
        """Initialize embedding pool - all instances are identical, no testing needed"""
        try:
            # OPTIMIZATION: Embedding instances don't need testing, create all at once
            for _ in range(self.pool_size):
                embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
                self._embedding_instances.put(embeddings)
            return True
        except Exception as e:
            logger.error(f"Embedding pool initialization failed: {e}")
            return False
    
    def get_neo4j_graph(self) -> Neo4jGraph:
        """Get Neo4j connection - FAST with shorter timeout"""
        try:
            return self._neo4j_graphs.get(timeout=5)  # Reduced from 10 seconds
        except Empty:
            logger.warning("Neo4j pool empty, creating on-demand connection")
            return Neo4jGraph(
                url=NEO4J_URI,
                username=NEO4J_USER,
                password=NEO4J_PASSWORD
            )
    
    def return_neo4j_graph(self, graph: Neo4jGraph):
        """Return Neo4j connection to pool"""
        try:
            self._neo4j_graphs.put_nowait(graph)
        except:
            pass  # Pool full, let it be garbage collected
    
    def get_llm(self) -> ChatOpenAI:
        """Get LLM instance - FAST with shorter timeout"""
        try:
            return self._llm_instances.get(timeout=5)  # Reduced from 10 seconds
        except Empty:
            logger.warning("LLM pool empty, creating on-demand instance")
            return ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0)
    
    def return_llm(self, llm: ChatOpenAI):
        """Return LLM instance to pool"""
        try:
            self._llm_instances.put_nowait(llm)
        except:
            pass
    
    def get_embeddings(self) -> OpenAIEmbeddings:
        """Get embeddings instance - FAST with shorter timeout"""
        try:
            return self._embedding_instances.get(timeout=5)  # Reduced from 10 seconds
        except Empty:
            logger.warning("Embeddings pool empty, creating on-demand instance")
            return OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    
    def return_embeddings(self, embeddings: OpenAIEmbeddings):
        """Return embeddings instance to pool"""
        try:
            self._embedding_instances.put_nowait(embeddings)
        except:
            pass

# Global connection pool - OPTIMIZED
_connection_pool = None
_pool_initialization_started = False

def get_connection_pool() -> OptimizedConnectionPool:
    """Get the global OPTIMIZED connection pool instance"""
    global _connection_pool
    if _connection_pool is None:
        _connection_pool = OptimizedConnectionPool()
    return _connection_pool

def start_early_pool_initialization():
    """Start pool initialization early (non-blocking) - SAFE to call multiple times"""
    global _connection_pool, _pool_initialization_started
    
    try:
        if not _pool_initialization_started:
            _pool_initialization_started = True
            logger.info("🚀 Starting EARLY pool initialization (user-triggered)")
            
            # Create pool if doesn't exist
            if _connection_pool is None:
                _connection_pool = OptimizedConnectionPool()
            
            # If pool thread died or never started, restart it
            if not hasattr(_connection_pool, '_initialization_thread') or \
               not _connection_pool._initialization_thread.is_alive():
                _connection_pool._initialization_thread = threading.Thread(
                    target=_connection_pool._fast_initialize_pools
                )
                _connection_pool._initialization_thread.daemon = True
                _connection_pool._initialization_thread.start()
                
            logger.info("✅ Early pool initialization started in background")
        else:
            logger.info("ℹ️  Pool initialization already started")
            
    except Exception as e:
        logger.warning(f"Early pool initialization failed, will use normal fallback: {e}")

def is_pool_ready() -> bool:
    """Check if pool is ready (non-blocking)"""
    global _connection_pool
    if _connection_pool is None:
        return False
    return getattr(_connection_pool, '_initialized', False)

def get_pool_status() -> Dict[str, Any]:
    """Get detailed pool status for debugging"""
    global _connection_pool, _pool_initialization_started
    
    if _connection_pool is None:
        return {"status": "not_created", "ready": False}
    
    return {
        "status": "ready" if _connection_pool._initialized else "initializing",
        "ready": _connection_pool._initialized,
        "started": _pool_initialization_started,
        "thread_alive": hasattr(_connection_pool, '_initialization_thread') and 
                       _connection_pool._initialization_thread.is_alive()
    }

# === OPTIMIZED INITIALIZATION ===
try:
    # Initialize with OPTIMIZED connection pool
    pool = get_connection_pool()
    
    # Get one instance of each for compatibility - NON-BLOCKING
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0)
    embedding_provider = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD
    )
    
    # OPTIMIZATION: Single quick test, don't block on it
    try:
        graph.query("RETURN 1 as test")
        logger.info("✅ Neo4j connection successful")
    except Exception as e:
        logger.warning(f"⚠️  Neo4j test failed, but continuing: {e}")

except Exception as e:
    logger.error(f"Component initialization failed: {e}")
    raise

# === OPTIMIZED PARALLEL PROCESSING ===

def optimized_batch_llm_calls(prompts: List[str], batch_size: int = BATCH_SIZE) -> List[str]:
    """OPTIMIZED: Execute LLM calls with better resource management and faster processing"""
    results = [""] * len(prompts)
    pool = get_connection_pool()
    
    def process_prompt(idx_prompt_pair):
        idx, prompt = idx_prompt_pair
        llm_instance = pool.get_llm()
        try:
            response = llm_instance.invoke(prompt)
            return idx, response.content
        except Exception as e:
            logger.error(f"Error in LLM call {idx}: {e}")
            return idx, f"Error: {str(e)}"
        finally:
            pool.return_llm(llm_instance)
    
    # OPTIMIZATION: More aggressive parallelization for faster answer generation
    max_workers = min(MAX_WORKERS, len(prompts), 8)  # Allow up to 8 parallel LLM calls
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_prompt, (i, prompt)) 
            for i, prompt in enumerate(prompts)
        ]
        
        # OPTIMIZATION: Shorter timeout for faster failure handling
        for future in as_completed(futures, timeout=20):  # Reduced from 30
            try:
                idx, result = future.result(timeout=10)  # Reduced from 15
                results[idx] = result
            except Exception as e:
                logger.error(f"Error in batch LLM processing: {e}")
    
    return results

def optimized_parallel_neo4j_retrieval(queries: List[str], course_id: str, 
                           results_per_query: int = 3,
                           use_enhanced_retrieval: bool = True) -> List[List[Document]]:
    """OPTIMIZED: Execute Neo4j retrievals with better performance"""
    results = [[] for _ in range(len(queries))]
    pool = get_connection_pool()
    
    # OPTIMIZATION: Pre-classify all queries to avoid doing it in threads
    query_types = [classify_query_type(query) for query in queries]
    
    def retrieve_for_query(idx_query_pair):
        idx, query = idx_query_pair
        query_type = query_types[idx]
        
        graph_instance = pool.get_neo4j_graph()
        embeddings_instance = pool.get_embeddings()
        
        try:
            # Get appropriate retrieval query
            is_list_query = query_type == "list"
            course_query = get_course_query(
                course_id,
                enhanced=use_enhanced_retrieval,
                list_query=is_list_query
            )
            
            # Create vector retriever
            chunk_vector = Neo4jVector.from_existing_index(
                embedding=embeddings_instance,
                graph=graph_instance,
                index_name=VECTOR_INDEX_NAME,
                embedding_node_property=EMBEDDING_PROPERTY,
                text_node_property="text",
                retrieval_query=course_query,
                node_label="Chunk"
            )
            
            # Retrieve documents - FIXED: Use invoke() instead of deprecated method
            chunk_retriever = chunk_vector.as_retriever(search_kwargs={"k": results_per_query})
            docs = chunk_retriever.invoke(query)
            
            return idx, docs
            
        except Exception as e:
            logger.error(f"Error in Neo4j retrieval for query {idx}: {e}")
            return idx, []
        finally:
            pool.return_neo4j_graph(graph_instance)
            pool.return_embeddings(embeddings_instance)
    
    # OPTIMIZATION: Use optimal worker count
    max_workers = min(MAX_WORKERS, len(queries), CONNECTION_POOL_SIZE)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(retrieve_for_query, (i, query)) 
            for i, query in enumerate(queries)
        ]
        
        # OPTIMIZATION: Shorter timeout for faster processing
        for future in as_completed(futures, timeout=25):  # Reduced from no timeout
            try:
                idx, docs = future.result(timeout=15)  # Reduced from 20
                results[idx] = docs
            except Exception as e:
                logger.error(f"Error in parallel retrieval: {e}")
    
    return results

# === OPTIMIZED CORE FUNCTIONS ===

def generate_diverse_queries(original_query: str, num_queries: int = 5) -> List[str]:
    """OPTIMIZED: Generate diverse queries with improved prompt and faster processing + caching"""
    
    # OPTIMIZATION: Simple caching for repeated queries
    cache_key = f"{original_query}_{num_queries}"
    if CACHE_ENABLED and cache_key in _query_cache:
        logger.info(f"Using cached diverse queries for: '{original_query}'")
        return _query_cache[cache_key]
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """
        Generate {num_queries} alternative queries for RAG retrieval. Be concise and focused.

        Rules:
        1. First query: grammatically corrected version of original
        2. Subsequent: different aspects, synonyms, sub-topics
        3. Include one broader generalization
        4. Include one about limitations/restrictions

        Return ONLY a JSON array of query strings.

        Example:
        "can i send 80lb pack with usps" → [
            "Can I send an 80 lb package with USPS?",
            "What is the maximum weight limit for USPS packages?",
            "Are there restrictions for heavy packages with USPS?",
            "How does USPS handle 80-pound packages?",
            "What shipping options exist for heavy USPS packages?"
        ]
        """),
        ("human", f"Original: {original_query}")
    ])

    try:
        pool = get_connection_pool()
        llm_instance = pool.get_llm()
        
        try:
            response = llm_instance.invoke(prompt_template.format_messages(num_queries=num_queries))
            queries = json.loads(response.content)
            
            # Ensure we have the right number of queries
            if len(queries) < num_queries:
                queries.extend([original_query] * (num_queries - len(queries)))
            elif len(queries) > num_queries:
                queries = queries[:num_queries]

            # Cache the result
            if CACHE_ENABLED:
                _query_cache[cache_key] = queries

            logger.info(f"Generated {len(queries)} diverse queries for: '{original_query}'")
            return queries
            
        finally:
            pool.return_llm(llm_instance)
            
    except Exception as e:
        logger.error(f"Error generating diverse queries: {e}")
        # Fallback with simpler variations
        fallback_queries = [original_query]
        for i in range(num_queries - 1):
            fallback_queries.append(f"How to {original_query.lower()}")
        return fallback_queries

def enhance_query_with_llm(original_query: str, course_id: str) -> str:
    """OPTIMIZED: Enhanced query improvement with connection pooling"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a query enhancement system for an e-learning knowledge graph system backed by Neo4J + vector search. 
        Transform user queries into more effective questions.

        Guidelines:
        1. If the query is well-formed (starts with "What is", "How do I", etc.), make minimal changes
        2. If the query is keywords/short phrase, transform it into a complete question
        3. For list queries (e.g., "B's of customer service"), ensure you clearly ask for the complete list
        4. Return ONLY the enhanced query without explanations

        Examples:
        - "Form 1583" -> "What is Form 1583 and how is it used?"
        - "fedex ground delivery" -> "What days does FedEx Ground deliver packages?"
        - "B's of customer service" -> "What are all the B's of great customer service and what does each one mean?"
        """),
        ("human", f"Original query: {original_query}")
    ])

    try:
        pool = get_connection_pool()
        llm_instance = pool.get_llm()
        
        try:
            response = llm_instance.invoke(prompt.format_messages())
            enhanced_query = response.content.strip()
            logger.info(f"Query enhancement: '{original_query}' -> '{enhanced_query}'")
            return enhanced_query
        finally:
            pool.return_llm(llm_instance)
            
    except Exception as e:
        logger.error(f"Error enhancing query: {e}")
        return original_query

def optimized_multi_query_retrieval_with_individual_answers(
        course_id: str,
        original_question: str,
        num_queries: int = 10,
        results_per_query: int = 3,
        use_enhanced_retrieval: bool = True,
        debug_mode: bool = False
) -> Dict[str, Any]:
    """
    BLAZING FAST: Ultra-optimized multi-query retrieval with adaptive performance
    """
    try:
        start_time = time.time()
        
        # OPTIMIZATION: Adaptive query count based on system performance
        pool = get_connection_pool()
        if not pool._initialized:
            logger.info("Pool not fully initialized, reducing query count for faster processing")
            num_queries = min(num_queries, 6)  # Reduce load when pool isn't ready
        
        # Step 1: Generate diverse queries (single LLM call)
        diverse_queries = generate_diverse_queries(original_question, num_queries)
        query_gen_time = time.time() - start_time
        logger.info(f"Generated {len(diverse_queries)} queries in {query_gen_time:.2f}s")
        
        # Step 2: OPTIMIZED parallel Neo4j retrieval for all queries
        retrieval_start = time.time()
        max_parallel_workers = min(MAX_WORKERS, len(diverse_queries), CONNECTION_POOL_SIZE)
        logger.info(f"Starting parallel retrieval with {max_parallel_workers} workers and {CONNECTION_POOL_SIZE} connections")
        all_docs_lists = optimized_parallel_neo4j_retrieval(
            diverse_queries, course_id, results_per_query, use_enhanced_retrieval
        )
        retrieval_time = time.time() - retrieval_start
        successful_retrievals = len([docs for docs in all_docs_lists if docs])
        logger.info(f"Completed parallel retrieval in {retrieval_time:.2f}s ({successful_retrievals}/{len(diverse_queries)} successful)")
        
        # Step 3: Prepare prompts for parallel answer generation
        answer_prompt_template = ChatPromptTemplate.from_messages([
            ("system", """
            You are an intelligent assistant for an e-learning platform called LearnChain. 
            Use the given context to answer the question clearly and concisely.

            If the context doesn't contain the answer, say: 'I don't have enough information 
            in the course materials to answer that question.'

            Context: {context}
            """),
            ("human", "{question}")
        ])
        
        # Prepare all prompts for batch processing
        prompts = []
        valid_queries = []
        
        for i, (query, docs) in enumerate(zip(diverse_queries, all_docs_lists)):
            if docs:  # Only process queries that returned documents
                combined_text = "\n\n".join([doc.page_content for doc in docs])
                prompt = answer_prompt_template.format_messages(
                    context=combined_text,
                    question=query
                )
                prompts.append(prompt[1].content)  # Get the formatted prompt text
                valid_queries.append((i, query, docs))
            else:
                logger.info(f"No documents retrieved for query: '{query}'")
        
        # Step 4: OPTIMIZED parallel answer generation
        answer_start = time.time()
        if prompts:
            # OPTIMIZATION: Reduce worker count if we have few prompts
            worker_count = min(len(prompts), MAX_WORKERS, 8)
            logger.info(f"Starting parallel LLM processing for {len(prompts)} prompts with {worker_count} workers")
            batch_responses = optimized_batch_llm_calls(prompts, batch_size=BATCH_SIZE)
        else:
            batch_responses = []
        answer_time = time.time() - answer_start
        logger.info(f"Generated {len(batch_responses)} answers in {answer_time:.2f}s (avg: {answer_time/max(1,len(batch_responses)):.2f}s per answer)")
        
        # Step 5: Compile query results
        query_results = []
        response_idx = 0
        
        for i, query in enumerate(diverse_queries):
            # Find if this query had valid results
            valid_entry = next((entry for entry in valid_queries if entry[0] == i), None)
            
            if valid_entry:
                _, _, docs = valid_entry
                answer = batch_responses[response_idx] if response_idx < len(batch_responses) else "Error generating answer"
                sources = list({doc.metadata.get("doc_name", "Unknown") for doc in docs})
                has_info = "I don't have enough information" not in answer
                response_idx += 1
            else:
                answer = "I don't have enough information in the course materials to answer that question."
                sources = []
                has_info = False
            
            query_results.append({
                "query": query,
                "answer": answer,
                "sources": sources,
                "has_info": has_info
            })
        
        # Step 6: Generate consolidated answer - OPTIMIZED
        consolidation_start = time.time()
        consolidated_answer = generate_consolidated_answer(original_question, query_results)
        consolidation_time = time.time() - consolidation_start
        
        # Compile results
        all_sources = []
        for result in query_results:
            all_sources.extend(result['sources'])
        unique_sources = list(set(all_sources))
        
        query_info = {
            "original_query": original_question,
            "enhanced_query": diverse_queries[0] if diverse_queries else original_question,
            "diverse_queries": diverse_queries[1:] if len(diverse_queries) > 1 else [],
            "query_type": classify_query_type(original_question)
        }
        
        total_time = time.time() - start_time
        logger.info(f"🚀 BLAZING FAST multi-query retrieval completed in {total_time:.2f}s "
                   f"(queries: {query_gen_time:.2f}s, retrieval: {retrieval_time:.2f}s, "
                   f"answers: {answer_time:.2f}s, consolidation: {consolidation_time:.2f}s)")
        
        return {
            "answer": consolidated_answer,
            "query_info": query_info,
            "source_documents": unique_sources,
            "individual_results": query_results,
            "performance_metrics": {
                "total_time": total_time,
                "query_generation_time": query_gen_time,
                "retrieval_time": retrieval_time,
                "answer_generation_time": answer_time,
                "consolidation_time": consolidation_time,
                "num_queries": len(diverse_queries),
                "num_valid_results": len([r for r in query_results if r['has_info']])
            },
            "raw_json": json.dumps({
                "original_question": original_question,
                "query_results": query_results,
                "consolidated_answer": consolidated_answer,
                "sources": unique_sources
            }, indent=2)
        }

    except Exception as e:
        logger.error(f"Error in optimized multi-query retrieval: {e}")
        return {
            "answer": f"Error retrieving information: {str(e)}",
            "query_info": {
                "original_query": original_question,
                "error": str(e)
            },
            "source_documents": [],
            "individual_results": [],
            "raw_json": json.dumps({"error": str(e)})
        }

def generate_consolidated_answer(original_question: str, query_results: List[Dict]) -> str:
    """Generate consolidated answer using connection pool - OPTIMIZED for faster processing"""
    consolidation_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are an intelligent assistant for LearnChain e-learning platform.

        Synthesize a comprehensive answer to the original question from the provided query results.
        Focus on accuracy and completeness. If multiple answers have useful information, combine them coherently.
        If no useful information is found, indicate insufficient information.

        Be concise and use bullet points for lists when appropriate.
        """),
        ("human", """
        Question: {original_question}

        Results:
        {query_results}

        Provide a consolidated answer.
        """)
    ])

    # OPTIMIZATION: More efficient result formatting
    formatted_results = ""
    useful_results = [r for r in query_results if r.get('has_info', False)]
    
    for i, result in enumerate(useful_results[:5]):  # Limit to top 5 for faster processing
        formatted_results += f"{i+1}. Q: {result['query']}\n   A: {result['answer']}\n\n"

    try:
        pool = get_connection_pool()
        llm_instance = pool.get_llm()
        
        try:
            response = llm_instance.invoke(
                consolidation_prompt.format_messages(
                    original_question=original_question,
                    query_results=formatted_results
                )
            )
            return response.content
        finally:
            pool.return_llm(llm_instance)
            
    except Exception as e:
        logger.error(f"Error generating consolidated answer: {e}")
        return f"Error consolidating answers: {str(e)}"

# === OPTIMIZED EXISTING FUNCTIONALITY ===

def query_course_knowledge(
        course_id: str,
        question: str,
        use_query_enhancement: bool = True,
        use_enhanced_retrieval: bool = True,
        k: int = DEFAULT_NUM_RESULTS,
        use_parallel_processing: bool = True
) -> Dict[str, Any]:
    """
    OPTIMIZED: Query processing with connection pooling and optional parallel processing
    """
    print("---------------------------------------------------------------", course_id)
    course_id = "Course_" + course_id

    try:
        start_time = time.time()
        
        # Step 1: Enhance the query if requested
        enhanced_question = question
        original_question = question

        if use_query_enhancement:
            try:
                enhanced_question = enhance_query_with_llm(question, course_id)
                logger.info(f"Enhanced query: '{question}' -> '{enhanced_question}'")
            except Exception as e:
                logger.error(f"Error enhancing query, using original: {e}")
                enhanced_question = question

        # Step 2: Classify query type
        query_type = classify_query_type(enhanced_question)
        logger.info(f"Query classified as: {query_type}")

        # Step 3: Get appropriate course-specific query
        is_list_query = query_type == "list"
        course_query = get_course_query(
            course_id,
            enhanced=use_enhanced_retrieval,
            list_query=is_list_query
        )

        # Step 4: Use OPTIMIZED connection pool for retrieval
        pool = get_connection_pool()
        graph_instance = pool.get_neo4j_graph()
        embeddings_instance = pool.get_embeddings()
        
        try:
            # Create vector retriever
            chunk_vector = Neo4jVector.from_existing_index(
                embedding=embeddings_instance,
                graph=graph_instance,
                index_name=VECTOR_INDEX_NAME,
                embedding_node_property=EMBEDDING_PROPERTY,
                text_node_property="text",
                retrieval_query=course_query,
                node_label="Chunk"
            )

            # Use more results for list queries
            search_k = k * 2 if is_list_query else k
            chunk_retriever = chunk_vector.as_retriever(search_kwargs={"k": search_k})
            
            # Create the chain using pooled LLM
            llm_instance = pool.get_llm()
            try:
                chunk_chain = create_stuff_documents_chain(llm_instance, prompt)
                retriever_chain = create_retrieval_chain(chunk_retriever, chunk_chain)
                
                # Execute query
                response = retriever_chain.invoke({"input": enhanced_question})
            finally:
                pool.return_llm(llm_instance)

        finally:
            pool.return_neo4j_graph(graph_instance)
            pool.return_embeddings(embeddings_instance)

        total_time = time.time() - start_time
        logger.info(f"Query completed in {total_time:.2f}s")
        logger.info(f"Retrieved {len(response.get('context', []))} documents")

        # Track if query was enhanced
        query_info = {}
        if use_query_enhancement and enhanced_question != original_question:
            query_info["original_query"] = original_question
            query_info["enhanced_query"] = enhanced_question
            query_info["query_type"] = query_type

        return {
            "answer": response.get("answer", ""),
            "context": response.get("context", []),
            "source_documents": list({doc.metadata.get("doc_name", "Unknown")
                                      for doc in response.get("context", [])
                                      }),
            "query_info": query_info,
            "performance_metrics": {
                "total_time": total_time,
                "enhanced": use_query_enhancement,
                "query_type": query_type
            }
        }
    except Exception as e:
        logger.error(f"Error querying course knowledge: {e}")
        return {
            "answer": f"Error querying knowledge graph: {str(e)}",
            "context": [],
            "source_documents": [],
            "query_info": {}
        }

# === OPTIMIZED HELPER FUNCTIONS ===

def check_apoc_availability(graph: Neo4jGraph) -> bool:
    try:
        result = graph.query("CALL apoc.help('') YIELD name RETURN name LIMIT 1")
        return True
    except Exception as e:
        logger.warning(f"APOC availability check failed: {e}")
        return False

# OPTIMIZATION: Only check APOC once at startup
APOC_AVAILABLE = check_apoc_availability(graph)
logger.info(f"APOC available: {APOC_AVAILABLE}")

def check_vector_index():
    """OPTIMIZED: Check if the vector index exists - faster check"""
    try:
        # OPTIMIZATION: Single quick test instead of multiple checks
        test_vector = [0.0] * 1536
        graph.query(
            f"""
            CALL db.index.vector.queryNodes('{VECTOR_INDEX_NAME}', $embedding, 1)
            YIELD node
            RETURN count(node) as count
            """,
            {"embedding": test_vector}
        )
        logger.info(f"Vector index '{VECTOR_INDEX_NAME}' exists")
        return True

    except Exception as e:
        logger.warning(f"Vector index '{VECTOR_INDEX_NAME}' not functional: {e}")
        return False

# Retrieval queries (unchanged - preserve working logic)
BASIC_RETRIEVAL_QUERY = """
MATCH (node)-[:PART_OF]->(d:Document)
WITH node, score, d
RETURN 
    node.text as text, 
    score,
    { 
        document: d.id,
        doc_name: d.name,
        course_id: d.course_id,
        user_id: d.user_id,
        org_id: d.org_id
    } AS metadata
"""

ENHANCED_RETRIEVAL_QUERY = """
MATCH (node)-[:PART_OF]->(d:Document)
WITH node, score, d
OPTIONAL MATCH (node)-[:HAS_ENTITY]->(e)
WITH node, score, d, collect(e.text) as entity_texts
RETURN 
    node.text as text, 
    score,
    { 
        document: d.id,
        doc_name: d.name,
        course_id: d.course_id,
        user_id: d.user_id,
        org_id: d.org_id,
        entities: entity_texts
    } AS metadata
ORDER BY score DESC
"""

LIST_RETRIEVAL_QUERY = """
MATCH (node)-[:PART_OF]->(d:Document)
WITH node, score, d
WITH node, score, d,
     CASE WHEN node.text CONTAINS '1.' OR 
               node.text CONTAINS '2.' OR 
               node.text CONTAINS '- ' OR 
               node.text CONTAINS '* ' 
          THEN true ELSE false END as contains_list
WITH node, 
     CASE WHEN contains_list THEN score * 1.2 ELSE score END as adjusted_score, 
     d, contains_list
RETURN 
    node.text as text, 
    adjusted_score as score,
    { 
        document: d.id,
        doc_name: d.name,
        course_id: d.course_id,
        user_id: d.user_id,
        org_id: d.org_id,
        contains_list: contains_list
    } AS metadata
ORDER BY adjusted_score DESC
"""

def get_course_query(course_id, enhanced=False, list_query=False):
    """Get a course-specific query with proper filtering."""
    if list_query:
        base_query = LIST_RETRIEVAL_QUERY
    elif enhanced:
        base_query = ENHANCED_RETRIEVAL_QUERY
    else:
        base_query = BASIC_RETRIEVAL_QUERY

    if not course_id:
        return base_query

    query_lines = base_query.split('\n')
    for i, line in enumerate(query_lines):
        if "WITH node," in line and "score" in line and "d" in line:
            query_lines.insert(i + 1, f"WHERE node.course_id = '{course_id}'")
            break

    return '\n'.join(query_lines)

instructions = (
    "You are an intelligent assistant for an e-learning platform called LearnChain. "
    "Use the given context to answer the question clearly and concisely.\n\n"
    "If the context doesn't contain the answer, say: 'I don't have enough information "
    "in the course materials to answer that question.'\n\n"
    "When referencing specific documents or entities in your response, use their names "
    "as they appear in the context.\n\n"
    "If the question asks about a list or series of items, make sure to include all items "
    "from the list if they're available in the context.\n\n"
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", instructions),
    ("human", "{input}")
])

# Create chunk chain with pooled LLM (will be recreated as needed)
chunk_chain = create_stuff_documents_chain(llm, prompt)

def classify_query_type(query: str) -> str:
    """Classify the query to determine the best retrieval strategy."""
    query_lower = query.lower()

    list_indicators = ["list", "all of the", "steps", "b's", "7 b", "seven b", "items", "points", "stages"]
    for indicator in list_indicators:
        if indicator in query_lower:
            return "list"

    complex_indicators = ["compare", "relationship", "difference", "connection", "integration", "why", "how"]
    for indicator in complex_indicators:
        if indicator in query_lower:
            return "complex"

    return "basic"

# Keep all display functions unchanged (preserve working logic)
def typewriter_effect(text: str, markdown: bool = True, delay_chars: float = CHAR_DELAY,
                      delay_words: float = WORD_DELAY) -> None:
    """Print text with a typewriter effect."""
    if delay_chars <= 0 and delay_words <= 0:
        if markdown:
            try:
                console.print(Markdown(text))
            except Exception as e:
                console.print(text)
                logging.warning(f"Error rendering markdown: {e}")
        else:
            console.print(text)
        return

    if markdown:
        rendered_text = ""
        with Live(console=console, refresh_per_second=20, transient=False) as live:
            for char in text:
                rendered_text += char
                try:
                    live.update(Markdown(rendered_text))
                except Exception:
                    live.update(Text(rendered_text))

                time.sleep(delay_chars)

                if char in " .,!?;:":
                    time.sleep(delay_words)
    else:
        rendered_text = ""
        for char in text:
            rendered_text += char
            console.print(rendered_text, end="\r")
            time.sleep(delay_chars)

            if char in " .,!?;:":
                time.sleep(delay_words)

        console.print()

def display_rich_answer(answer: str, query_info: Dict = None, sources: List[str] = None,
                        individual_results: List[Dict] = None, show_json: bool = False, raw_json: str = None):
    """Display the answer and metadata using rich formatting with typewriter effect."""
    try:
        os.system('cls' if os.name == 'nt' else 'clear')
    except:
        console.print("\n\n")

    if query_info and query_info.get("original_query"):
        console.print(Panel("[bold]Query Enhancement[/bold]", style="blue", width=terminal_width - 2))
        console.print(f"Original: [italic]{query_info['original_query']}[/italic]")

        if query_info.get("enhanced_query"):
            console.print(f"Enhanced: [bold]{query_info['enhanced_query']}[/bold]")

        if query_info.get("diverse_queries"):
            console.print("\n[bold]Additional Queries:[/bold]")
            for i, q in enumerate(query_info["diverse_queries"], 1):
                console.print(f"{i}. [cyan]{q}[/cyan]")

        if query_info.get("query_type"):
            console.print(f"Query type: [cyan]{query_info['query_type']}[/cyan]")

        console.print()

    console.print(Panel("[bold]Answer[/bold]", style="green", width=terminal_width - 2))

    try:
        typewriter_effect(answer, markdown=True)
    except Exception as e:
        logging.warning(f"Error rendering markdown with typewriter effect: {e}")
        typewriter_effect(answer, markdown=False)

    if sources:
        console.print(Panel("[bold]Sources[/bold]", style="yellow", width=terminal_width - 2))
        for source in sources:
            console.print(f"- [blue]{source}[/blue]")

    if individual_results and show_json:
        console.print(Panel("[bold]Individual Query Results[/bold]", style="magenta", width=terminal_width - 2))
        for i, result in enumerate(individual_results):
            console.print(f"[bold]Query {i + 1}:[/bold] {result['query']}")
            console.print(f"[bold]Answer:[/bold] {result['answer']}")
            console.print(f"[bold]Sources:[/bold] {', '.join(result['sources'])}")
            console.print(f"[bold]Has info:[/bold] {'Yes' if result['has_info'] else 'No'}")
            console.print("")

    if raw_json and show_json:
        console.print(Panel("[bold]Raw JSON[/bold]", style="cyan", width=terminal_width - 2))
        try:
            parsed_json = json.loads(raw_json)
            console.print(json.dumps(parsed_json, indent=2))
        except:
            console.print(raw_json)

def run_interactive_mode():
    """OPTIMIZED: Run the retriever in interactive mode with enhanced output."""
    global FAST_MODE, CHAR_DELAY, WORD_DELAY

    os.environ.setdefault("TERM", "xterm-256color")

    console.print(Panel("[bold]🚀 LearnChain Knowledge Graph Retriever - BLAZING FAST OPTIMIZED[/bold]",
                        style="bold green",
                        subtitle="Type 'exit' to quit | Optimized for maximum speed!",
                        width=terminal_width - 2))

    if not check_vector_index():
        console.print(Panel("[bold red]WARNING: Vector index not found![/bold red]", width=terminal_width - 2))
        console.print("Please run the graph builder script first to create the knowledge graph.")
        proceed = input("Do you want to continue anyway? (y/n): ").lower()
        if proceed != 'y':
            console.print("[red]Exiting. Please run the graph builder script first.[/red]")
            return

    default_course_id = "001"
    console.print()
    course_id = input(f"Enter course ID (default: {default_course_id}): ")
    if not course_id:
        course_id = default_course_id

    course_id = "Course_" + course_id
    console.print(f"Using course ID: [bold]{course_id}[/bold]")

    console.print()
    use_multi_query = input("Use multi-query retrieval? (y/n, default: y): ").lower() != "n"
    num_queries = 5
    if use_multi_query:
        try:
            num_queries_input = input(f"Number of queries to generate (default: {num_queries}): ")
            if num_queries_input:
                num_queries = int(num_queries_input)
        except ValueError:
            console.print(f"[yellow]Invalid number, using default: {num_queries}[/yellow]")

    use_enhanced_retrieval = input(
        "Use enhanced retrieval with graph relationships? (y/n, default: y): ").lower() != "n"

    show_json = input("Show detailed JSON results? (y/n, default: n): ").lower() == "y"
    show_performance = input("Show performance metrics? (y/n, default: y): ").lower() != "n"

    typewriter_speed = input("Typewriter effect speed (fast/medium/slow/off, default: fast): ").lower()
    if typewriter_speed == "off":
        FAST_MODE = False
        CHAR_DELAY = 0
        WORD_DELAY = 0
    elif typewriter_speed == "medium":
        FAST_MODE = False
        CHAR_DELAY = 0.01
        WORD_DELAY = 0.03
    elif typewriter_speed == "slow":
        FAST_MODE = False
        CHAR_DELAY = 0.02
        WORD_DELAY = 0.05
    else:
        FAST_MODE = True
        CHAR_DELAY = 0.005
        WORD_DELAY = 0.02

    status_text = Text()
    status_text.append("🚀 BLAZING FAST Features: ", style="bold green")
    status_text.append("Multi-query retrieval: ", style="bold")
    status_text.append(f"ENABLED ({num_queries} queries)\n" if use_multi_query else "DISABLED\n",
                       style="green" if use_multi_query else "red")
    status_text.append("Enhanced retrieval: ", style="bold")
    status_text.append("ENABLED\n" if use_enhanced_retrieval else "DISABLED\n",
                       style="green" if use_enhanced_retrieval else "red")
    status_text.append("⚡ Optimized parallel processing: ", style="bold")
    status_text.append(f"ENABLED ({MAX_WORKERS} workers, {CONNECTION_POOL_SIZE} pool)\n", style="green")
    status_text.append("🔥 Fast connection pooling: ", style="bold")
    status_text.append(f"ENABLED ({CONNECTION_POOL_SIZE} connections)\n", style="green")
    status_text.append("📊 Performance metrics: ", style="bold")
    status_text.append("ENABLED\n" if show_performance else "DISABLED\n",
                       style="green" if show_performance else "red")
    console.print(Panel(status_text, width=terminal_width - 2))

    while True:
        console.print()
        question = input("Enter your question (or 'exit' to quit): ")
        if question.lower() == "exit":
            break

        with console.status("[bold green]🚀 Processing query with BLAZING FAST optimization...[/bold green]"):
            if use_multi_query:
                result = optimized_multi_query_retrieval_with_individual_answers(
                    course_id,
                    question,
                    num_queries=num_queries,
                    use_enhanced_retrieval=use_enhanced_retrieval,
                    debug_mode=show_json
                )
            else:
                result = query_course_knowledge(
                    course_id,
                    question,
                    use_query_enhancement=True,
                    use_enhanced_retrieval=use_enhanced_retrieval
                )

        # Show performance metrics if enabled
        if show_performance and "performance_metrics" in result:
            metrics = result["performance_metrics"]
            console.print(Panel(f"⚡ [bold]BLAZING FAST Performance Metrics[/bold]\n"
                               f"🚀 Total time: {metrics.get('total_time', 0):.2f}s\n"
                               f"📝 Query generation: {metrics.get('query_generation_time', 0):.2f}s\n"
                               f"🔍 Retrieval: {metrics.get('retrieval_time', 0):.2f}s\n"
                               f"🤖 Answer generation: {metrics.get('answer_generation_time', 0):.2f}s\n"
                               f"🔗 Consolidation: {metrics.get('consolidation_time', 0):.2f}s",
                               style="blue", width=terminal_width - 2))

        display_rich_answer(
            result["answer"],
            result.get("query_info"),
            result.get("source_documents"),
            result.get("individual_results"),
            show_json=show_json,
            raw_json=result.get("raw_json")
        )

def get_retriever_answer(question: str, course_id: str,
                         use_query_enhancement: bool = True,
                         use_enhanced_retrieval: bool = True,
                         use_multi_query: bool = True,
                         num_queries: int = 5) -> Dict[str, Any]:
    """
    🚀 BLAZING FAST: Get an answer from the knowledge graph with optimized parallel processing.
    """
    print("---------------------------------------------------------------", course_id)
    course_id = "Course_" + course_id
    try:
        if use_multi_query:
            result = optimized_multi_query_retrieval_with_individual_answers(
                course_id,
                question,
                num_queries=num_queries,
                use_enhanced_retrieval=use_enhanced_retrieval
            )
        else:
            result = query_course_knowledge(
                course_id,
                question,
                use_query_enhancement=use_query_enhancement,
                use_enhanced_retrieval=use_enhanced_retrieval
            )

        return result
    except Exception as e:
        logger.error(f"Error in get_retriever_answer: {e}")
        return {
            "answer": f"Error retrieving information: {str(e)}",
            "context": [],
            "source_documents": [],
            "query_info": {},
            "individual_results": [],
            "raw_json": json.dumps({"error": str(e)})
        }

if __name__ == "__main__":
    run_interactive_mode()