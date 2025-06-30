# retriever.py - Clean version without logging

import os
from typing import Dict, Any, List, Optional, Tuple, Union
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
import time
import json
from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue, Empty
import dotenv
import atexit
import html


dotenv.load_dotenv()

# === CONFIG ===
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Performance settings
DEFAULT_NUM_RESULTS = 20
MAX_QUERY_TOKENS = 15000
VECTOR_INDEX_NAME = "chunkVector"
EMBEDDING_PROPERTY = "textEmbedding"
MAX_WORKERS = 8
BATCH_SIZE = 8
CONNECTION_POOL_SIZE = 8

# Store credentials in environment
os.environ["NEO4J_URI"] = NEO4J_URI
os.environ["NEO4J_USER"] = NEO4J_USER
os.environ["NEO4J_PASSWORD"] = NEO4J_PASSWORD
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# === CONNECTION POOL ===
class UltraFastConnectionPool:
    def __init__(self, pool_size: int = CONNECTION_POOL_SIZE):
        self.pool_size = pool_size
        self._neo4j_graphs = Queue(maxsize=pool_size)
        self._llm_instances = Queue(maxsize=pool_size)
        self._embedding_instances = Queue(maxsize=pool_size)
        self._lock = threading.Lock()
        self._initialized = False
        self._initialization_complete = threading.Event()
        self._start_preemptive_initialization()
    
    def _start_preemptive_initialization(self):
        self._initialization_thread = threading.Thread(target=self._preemptive_initialize)
        self._initialization_thread.daemon = False
        self._initialization_thread.start()
    
    def _preemptive_initialize(self):
        try:
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=3) as executor:
                neo4j_future = executor.submit(self._preemptive_init_neo4j)
                llm_future = executor.submit(self._preemptive_init_llm)
                embedding_future = executor.submit(self._preemptive_init_embedding)
                
                neo4j_success = neo4j_future.result(timeout=30)
                llm_success = llm_future.result(timeout=20)
                embedding_success = embedding_future.result(timeout=20)
                
                if neo4j_success and llm_success and embedding_success:
                    self._initialized = True
            
            self._initialization_complete.set()
        except Exception:
            self._initialization_complete.set()
    
    def _preemptive_init_neo4j(self) -> bool:
        try:
            test_graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD)
            test_graph.query("RETURN 1 as test")
            
            try:
                test_graph.query("CALL apoc.help('') YIELD name RETURN name LIMIT 1")
            except:
                pass
            
            try:
                test_result = test_graph.query(
                    f"SHOW INDEXES YIELD name WHERE name = '{VECTOR_INDEX_NAME}' RETURN count(*) as count"
                )
                if test_result and test_result[0]['count'] > 0:
                    try:
                        test_embedding = [0.0] * 1536
                        test_graph.query(
                            f"CALL db.index.vector.queryNodes('{VECTOR_INDEX_NAME}', $embedding, 1) YIELD node RETURN count(node) as count",
                            {"embedding": test_embedding}
                        )
                    except:
                        pass
            except:
                pass
            
            self._neo4j_graphs.put(test_graph)
            for i in range(self.pool_size - 1):
                graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD)
                self._neo4j_graphs.put(graph)
            
            return True
        except:
            return False
    
    def _preemptive_init_llm(self) -> bool:
        try:
            test_llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0)
            test_llm.invoke("Test connection")
            self._llm_instances.put(test_llm)
            
            for i in range(self.pool_size - 1):
                llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0)
                self._llm_instances.put(llm)
            
            return True
        except:
            return False
    
    def _preemptive_init_embedding(self) -> bool:
        try:
            test_embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
            test_embeddings.embed_query("test")
            self._embedding_instances.put(test_embeddings)
            
            for i in range(self.pool_size - 1):
                embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
                self._embedding_instances.put(embeddings)
            
            return True
        except:
            return False
    
    def wait_for_initialization(self, timeout: float = 30.0) -> bool:
        result = self._initialization_complete.wait(timeout)
        return result and self._initialized
    
    def is_ready(self) -> bool:
        return self._initialized
    
    def get_neo4j_graph(self) -> Neo4jGraph:
        try:
            return self._neo4j_graphs.get(timeout=2)
        except Empty:
            return Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD)
    
    def return_neo4j_graph(self, graph: Neo4jGraph):
        try:
            self._neo4j_graphs.put_nowait(graph)
        except:
            pass
    
    def get_llm(self) -> ChatOpenAI:
        try:
            return self._llm_instances.get(timeout=2)
        except Empty:
            return ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0)
    
    def return_llm(self, llm: ChatOpenAI):
        try:
            self._llm_instances.put_nowait(llm)
        except:
            pass
    
    def get_embeddings(self) -> OpenAIEmbeddings:
        try:
            return self._embedding_instances.get(timeout=2)
        except Empty:
            return OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    
    def return_embeddings(self, embeddings: OpenAIEmbeddings):
        try:
            self._embedding_instances.put_nowait(embeddings)
        except:
            pass

# Global connection pool
_connection_pool = UltraFastConnectionPool()

def get_connection_pool() -> UltraFastConnectionPool:
    return _connection_pool

def ensure_preemptive_ready(timeout: float = 30.0) -> bool:
    return _connection_pool.wait_for_initialization(timeout)

# === CLEANUP ===
def cleanup_connections():
    global _connection_pool
    if _connection_pool:
        time.sleep(0.5)

atexit.register(cleanup_connections)

# === PARALLEL PROCESSING ===
def ultra_fast_batch_llm_calls(prompts: List[str]) -> List[str]:
    results = [""] * len(prompts)
    pool = get_connection_pool()
    
    def process_prompt(idx_prompt_pair):
        idx, prompt = idx_prompt_pair
        llm_instance = pool.get_llm()
        try:
            response = llm_instance.invoke(prompt)
            return idx, response.content
        except Exception:
            return idx, ""
        finally:
            pool.return_llm(llm_instance)
    
    max_workers = min(MAX_WORKERS, len(prompts))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_prompt, (i, prompt)) 
            for i, prompt in enumerate(prompts)
        ]
        
        for future in as_completed(futures, timeout=25):
            try:
                idx, result = future.result(timeout=12)
                results[idx] = result
            except:
                pass
    
    return results

def ultra_fast_parallel_neo4j_retrieval(queries: List[str], course_id: str, 
                                       results_per_query: int = 3,
                                       use_enhanced_retrieval: bool = True) -> List[List[Document]]:
    results = [[] for _ in range(len(queries))]
    pool = get_connection_pool()
    
    # Batch embedding generation
    embeddings_instance = pool.get_embeddings()
    
    try:
        all_embeddings = embeddings_instance.embed_documents(queries)
    except:
        all_embeddings = []
        for query in queries:
            try:
                embedding = embeddings_instance.embed_query(query)
                all_embeddings.append(embedding)
            except:
                all_embeddings.append([0.0] * 1536)
    finally:
        pool.return_embeddings(embeddings_instance)
    
    query_types = [classify_query_type(query) for query in queries]
    
    def retrieve_for_query(idx_query_embedding_tuple):
        idx, query, query_embedding = idx_query_embedding_tuple
        query_type = query_types[idx]
        
        graph_instance = pool.get_neo4j_graph()
        
        try:
            is_list_query = query_type == "list"
            course_query = get_course_query(course_id, enhanced=use_enhanced_retrieval, list_query=is_list_query)
            
            class PrecomputedEmbeddings:
                def __init__(self, embedding):
                    self.embedding = embedding
                
                def embed_query(self, text):
                    return self.embedding
                
                def embed_documents(self, texts):
                    return [self.embedding] * len(texts)
            
            precomputed_embeddings = PrecomputedEmbeddings(query_embedding)
            
            chunk_vector = Neo4jVector.from_existing_index(
                embedding=precomputed_embeddings,
                graph=graph_instance,
                index_name=VECTOR_INDEX_NAME,
                embedding_node_property=EMBEDDING_PROPERTY,
                text_node_property="text",
                retrieval_query=course_query,
                node_label="Chunk"
            )
            
            chunk_retriever = chunk_vector.as_retriever(search_kwargs={"k": results_per_query})
            docs = chunk_retriever.invoke(query)
            
            return idx, docs
        except:
            return idx, []
        finally:
            pool.return_neo4j_graph(graph_instance)
    
    max_workers = min(MAX_WORKERS, len(queries))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(retrieve_for_query, (i, query, embedding)) 
            for i, (query, embedding) in enumerate(zip(queries, all_embeddings))
        ]
        
        for future in as_completed(futures, timeout=30):
            try:
                idx, docs = future.result(timeout=20)
                results[idx] = docs
            except:
                pass
    
    return results

# === CORE FUNCTIONS ===
def generate_diverse_queries(original_query: str, num_queries: int = 5) -> List[str]:
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """
        Generate {num_queries} alternative queries for RAG retrieval. Be concise and focused.

        Rules:
        1. First query: grammatically corrected version of original
        2. Subsequent: different aspects, synonyms, sub-topics, keywords
        3. Include one broader generalization
        4. Include one about limitations/restrictions

        Return ONLY a JSON array of query strings.
        """),
        ("human", f"Original: {original_query}")
    ])

    try:
        pool = get_connection_pool()
        llm_instance = pool.get_llm()
        
        try:
            response = llm_instance.invoke(prompt_template.format_messages(num_queries=num_queries))
            queries = json.loads(response.content)
            
            if len(queries) < num_queries:
                queries.extend([original_query] * (num_queries - len(queries)))
            elif len(queries) > num_queries:
                queries = queries[:num_queries]

            return queries
        finally:
            pool.return_llm(llm_instance)
    except:
        fallback_queries = [original_query]
        for i in range(num_queries - 1):
            fallback_queries.append(f"How to {original_query.lower()}")
        return fallback_queries

def ultra_fast_multi_query_retrieval(
        course_id: str,
        original_question: str,
        num_queries: int = 5,
        results_per_query: int = 3,
        use_enhanced_retrieval: bool = True
) -> Dict[str, Any]:
    try:
        start_time = time.time()
        
        pool = get_connection_pool()
        if not pool.is_ready():
            pool.wait_for_initialization(timeout=10)
            if not pool.is_ready():
                num_queries = min(num_queries, 6)
        
        # Generate diverse queries
        diverse_queries = generate_diverse_queries(original_question, num_queries)
        query_gen_time = time.time() - start_time
        
        # Parallel retrieval
        retrieval_start = time.time()
        all_docs_lists = ultra_fast_parallel_neo4j_retrieval(
            diverse_queries, course_id, results_per_query, use_enhanced_retrieval
        )
        retrieval_time = time.time() - retrieval_start
        successful_retrievals = len([docs for docs in all_docs_lists if docs])
        
        # Prepare prompts for parallel answer generation
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
        
        prompts = []
        valid_queries = []
        
        for i, (query, docs) in enumerate(zip(diverse_queries, all_docs_lists)):
            if docs:
                combined_text = "\n\n".join([doc.page_content for doc in docs])
                prompt = answer_prompt_template.format_messages(
                    context=combined_text,
                    question=query
                )
                prompts.append(prompt[1].content)
                valid_queries.append((i, query, docs))
        
        # Parallel answer generation
        answer_start = time.time()
        if prompts:
            batch_responses = ultra_fast_batch_llm_calls(prompts)
        else:
            batch_responses = []
        answer_time = time.time() - answer_start
        
        # Compile results
        query_results = []
        response_idx = 0
        
        for i, query in enumerate(diverse_queries):
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
        
        # Generate consolidated answer
        consolidation_start = time.time()
        consolidated_answer = generate_consolidated_answer(original_question, query_results)
        consolidation_time = time.time() - consolidation_start
        
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
            }
        }

    except Exception:
        return {
            "answer": f"Error retrieving information",
            "query_info": {"original_query": original_question},
            "source_documents": [],
            "individual_results": []
        }

def generate_consolidated_answer(original_question: str, query_results: List[Dict]) -> str:
    consolidation_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are an intelligent assistant for LearnChain e-learning platform.
        Synthesize a comprehensive answer to the original question from the provided query results.
        Focus on accuracy and completeness. If multiple answers have useful information, combine them coherently.
        Be concise and use bullet points for lists when appropriate.
        """),
        ("human", """
        Question: {original_question}
        Results: {query_results}
        Provide a consolidated answer.
        """)
    ])

    formatted_results = ""
    useful_results = [r for r in query_results if r.get('has_info', False)]
    
    # Skip consolidation if only one good result
    if len(useful_results) <= 1:
        if useful_results:
            return useful_results[0]['answer']
        else:
            return "I don't have enough information in the course materials to answer that question."
    
    for i, result in enumerate(useful_results[:3]):
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
    except:
        if useful_results:
            return useful_results[0]['answer']
        return "Error consolidating answers"

# === COMPATIBILITY FUNCTIONS ===
def query_course_knowledge(
        course_id: str,
        question: str,
        use_query_enhancement: bool = True,
        use_enhanced_retrieval: bool = True,
        k: int = DEFAULT_NUM_RESULTS,
        use_parallel_processing: bool = True
) -> Dict[str, Any]:
    course_id = "Course_" + course_id

    try:
        start_time = time.time()
        
        if use_parallel_processing:
            return ultra_fast_multi_query_retrieval(
                course_id=course_id,
                original_question=question,
                num_queries=5,
                results_per_query=max(k//10, 3),
                use_enhanced_retrieval=use_enhanced_retrieval
            )
        
        enhanced_question = question
        if use_query_enhancement:
            enhanced_question = enhance_query_with_llm(question, course_id)

        query_type = classify_query_type(enhanced_question)
        is_list_query = query_type == "list"
        course_query = get_course_query(course_id, enhanced=use_enhanced_retrieval, list_query=is_list_query)

        pool = get_connection_pool()
        graph_instance = pool.get_neo4j_graph()
        embeddings_instance = pool.get_embeddings()
        
        try:
            chunk_vector = Neo4jVector.from_existing_index(
                embedding=embeddings_instance,
                graph=graph_instance,
                index_name=VECTOR_INDEX_NAME,
                embedding_node_property=EMBEDDING_PROPERTY,
                text_node_property="text",
                retrieval_query=course_query,
                node_label="Chunk"
            )

            search_k = k * 2 if is_list_query else k
            chunk_retriever = chunk_vector.as_retriever(search_kwargs={"k": search_k})
            
            llm_instance = pool.get_llm()
            try:
                chunk_chain = create_stuff_documents_chain(llm_instance, prompt)
                retriever_chain = create_retrieval_chain(chunk_retriever, chunk_chain)
                response = retriever_chain.invoke({"input": enhanced_question})
            finally:
                pool.return_llm(llm_instance)

        finally:
            pool.return_neo4j_graph(graph_instance)
            pool.return_embeddings(embeddings_instance)

        total_time = time.time() - start_time

        query_info = {}
        if use_query_enhancement and enhanced_question != question:
            query_info["original_query"] = question
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
    except:
        return {
            "answer": "Error querying knowledge graph",
            "context": [],
            "source_documents": [],
            "query_info": {}
        }

def enhance_query_with_llm(original_query: str, course_id: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        Transform user queries into more effective questions for knowledge retrieval.
        Guidelines:
        1. If well-formed, make minimal changes
        2. If keywords/short phrase, transform to complete question
        3. For list queries, ensure you ask for the complete list
        4. Return ONLY the enhanced query
        """),
        ("human", f"Original query: {original_query}")
    ])

    try:
        pool = get_connection_pool()
        llm_instance = pool.get_llm()
        
        try:
            response = llm_instance.invoke(prompt.format_messages())
            enhanced_query = response.content.strip()
            return enhanced_query
        finally:
            pool.return_llm(llm_instance)
    except:
        return original_query

def get_retriever_answer(question: str, course_id: str,
                         use_query_enhancement: bool = True,
                         use_enhanced_retrieval: bool = True,
                         use_multi_query: bool = True,
                         num_queries: int = 5) -> Dict[str, Any]:
    course_id = "Course_" + course_id
    try:
        if use_multi_query:
            result = ultra_fast_multi_query_retrieval(
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
    except:
        return {
            "answer": "Error retrieving information",
            "context": [],
            "source_documents": [],
            "query_info": {},
            "individual_results": []
        }

# === HELPER FUNCTIONS ===
def classify_query_type(query: str) -> str:
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

# === RETRIEVAL QUERIES ===
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

# === PROMPT TEMPLATE ===
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

# === INITIALIZATION STATUS ===
def get_initialization_status() -> Dict[str, Any]:
    pool = get_connection_pool()
    return {
        "preemptive_ready": pool.is_ready(),
        "pool_size": pool.pool_size,
        "initialization_complete": pool._initialization_complete.is_set()
    }

def wait_for_full_initialization(timeout: float = 30.0) -> bool:
    return ensure_preemptive_ready(timeout)