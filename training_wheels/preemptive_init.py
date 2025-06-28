# preemptive_init.py - Preemptive initialization for blazing fast performance

import threading
import logging
import time
from typing import Optional, Callable, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class PreemptiveInitializer:
    """Handles all initialization tasks before user needs them"""
    
    def __init__(self):
        self.initialized = False
        self.initialization_thread = None
        self.pool = None
        self.neo4j_ready = False
        self.openai_ready = False
        self.embeddings_warmed = False
        self.start_time = None
        self.completion_callbacks = []
        self.query_cache = {}
        
    def start_background_init(self, on_complete: Optional[Callable] = None):
        """Start all initialization in background"""
        if self.initialization_thread and self.initialization_thread.is_alive():
            logger.info("Initialization already in progress")
            if on_complete:
                self.completion_callbacks.append(on_complete)
            return
            
        if on_complete:
            self.completion_callbacks.append(on_complete)
            
        self.initialization_thread = threading.Thread(target=self._run_initialization)
        self.initialization_thread.daemon = True
        self.initialization_thread.start()
        logger.info("🚀 Started preemptive initialization in background")
        
    def _run_initialization(self):
        """Run all initialization tasks"""
        self.start_time = time.time()
        
        try:
            # Step 1: Initialize connection pool immediately
            logger.info("⚡ Step 1: Initializing connection pool...")
            from retriever import ConnectionPool, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, OPENAI_API_KEY
            
            # Create a new pool that we'll manage
            self.pool = ConnectionPool()
            
            # Wait for pool initialization to complete
            if not self.pool.wait_for_init(timeout=10):
                logger.warning("Connection pool initialization timed out")
            
            # Step 2: Test Neo4j connection
            logger.info("⚡ Step 2: Testing Neo4j connection...")
            self._test_neo4j()
            
            # Step 3: Test OpenAI connection
            logger.info("⚡ Step 3: Testing OpenAI connection...")
            self._test_openai()
            
            # Step 4: Warm up embeddings
            logger.info("⚡ Step 4: Warming up embeddings...")
            self._warm_embeddings()
            
            # Step 5: Initialize other services
            logger.info("⚡ Step 5: Initializing other services...")
            self._init_other_services()
            
            # Step 6: Precompute common queries
            logger.info("⚡ Step 6: Precomputing common query embeddings...")
            self._precompute_common_queries()
            
            self.initialized = True
            elapsed = time.time() - self.start_time
            logger.info(f"✅ Preemptive initialization complete in {elapsed:.2f}s")
            
            # Call completion callbacks
            for callback in self.completion_callbacks:
                try:
                    callback(True, elapsed)
                except Exception as e:
                    logger.error(f"Error in completion callback: {e}")
                    
        except Exception as e:
            logger.error(f"❌ Preemptive initialization failed: {e}")
            self.initialized = False
            
            # Call completion callbacks with failure
            for callback in self.completion_callbacks:
                try:
                    callback(False, 0)
                except Exception as e:
                    logger.error(f"Error in completion callback: {e}")
    
    def _test_neo4j(self):
        """Test Neo4j connection and warm it up"""
        try:
            neo4j = self.pool.get_neo4j_graph()
            try:
                # Run a simple query to warm up connection
                result = neo4j.query("RETURN 1 as test")
                
                # Check if APOC is available
                try:
                    neo4j.query("RETURN apoc.version() as version")
                    logger.info("✅ APOC available")
                except:
                    logger.warning("⚠️ APOC not available")
                
                # Check vector index
                try:
                    neo4j.query("""
                        SHOW INDEXES
                        YIELD name, type
                        WHERE type = 'VECTOR'
                        RETURN count(*) as vector_indexes
                    """)
                    logger.info("✅ Vector indexes checked")
                except:
                    logger.warning("⚠️ Could not check vector indexes")
                    
                self.neo4j_ready = True
                logger.info("✅ Neo4j connection ready")
                
            finally:
                self.pool.return_neo4j_graph(neo4j)
                
        except Exception as e:
            logger.error(f"Neo4j test failed: {e}")
            self.neo4j_ready = False
    
    def _test_openai(self):
        """Test OpenAI connection"""
        try:
            llm = self.pool.get_llm()
            try:
                # Simple test query
                response = llm.invoke("Return 'OK' if you can read this")
                if response and response.content:
                    self.openai_ready = True
                    logger.info("✅ OpenAI connection ready")
                    
            finally:
                self.pool.return_llm(llm)
                
        except Exception as e:
            logger.error(f"OpenAI test failed: {e}")
            self.openai_ready = False
    
    def _warm_embeddings(self):
        """Warm up the embeddings model"""
        try:
            embeddings = self.pool.get_embeddings()
            try:
                # Generate a test embedding
                test_text = "This is a test query for warming up embeddings"
                result = embeddings.embed_query(test_text)
                if result and len(result) > 0:
                    self.embeddings_warmed = True
                    logger.info("✅ Embeddings model warmed up")
                    
            finally:
                self.pool.return_embeddings(embeddings)
                
        except Exception as e:
            logger.error(f"Embeddings warm-up failed: {e}")
            self.embeddings_warmed = False
    
    def _init_other_services(self):
        """Initialize other services like S3"""
        try:
            # Test S3 if enabled
            from training_wheels import Config, S3Manager
            if Config.S3_ENABLED:
                s3_manager = S3Manager()
                if s3_manager.enabled:
                    logger.info("✅ S3 ready")
                    
        except Exception as e:
            logger.error(f"Other services initialization failed: {e}")
    
    def _precompute_common_queries(self):
        """Precompute embeddings for common queries"""
        common_goals = [
            "create an aws ec2 instance",
            "deploy django on lightsail",
            "set up s3 bucket",
            "configure cloudfront",
            "create lambda function",
            "set up rds database",
            "configure api gateway",
            "create docker container",
            "deploy react app",
            "set up kubernetes cluster"
        ]
        
        try:
            embeddings_model = self.pool.get_embeddings()
            try:
                for goal in common_goals:
                    # Generate embedding
                    embedding = embeddings_model.embed_query(goal)
                    self.query_cache[goal] = {
                        "embedding": embedding,
                        "timestamp": time.time()
                    }
                    
                logger.info(f"✅ Precomputed {len(self.query_cache)} query embeddings")
                
            finally:
                self.pool.return_embeddings(embeddings_model)
                
        except Exception as e:
            logger.error(f"Failed to precompute embeddings: {e}")
    
    def wait_for_init(self, timeout: float = 30) -> bool:
        """Wait for initialization to complete"""
        if self.initialized:
            return True
            
        if not self.initialization_thread or not self.initialization_thread.is_alive():
            return False
            
        self.initialization_thread.join(timeout)
        return self.initialized
    
    def get_status(self) -> Dict[str, Any]:
        """Get current initialization status"""
        status = {
            "initialized": self.initialized,
            "neo4j_ready": self.neo4j_ready,
            "openai_ready": self.openai_ready,
            "embeddings_warmed": self.embeddings_warmed,
            "in_progress": self.initialization_thread and self.initialization_thread.is_alive(),
            "cached_queries": len(self.query_cache)
        }
        
        if self.start_time:
            elapsed = time.time() - self.start_time
            status["elapsed_time"] = elapsed
            
        return status

# Global instance
_preemptive_init = None

def get_preemptive_init() -> PreemptiveInitializer:
    """Get the global preemptive initializer"""
    global _preemptive_init
    if _preemptive_init is None:
        _preemptive_init = PreemptiveInitializer()
    return _preemptive_init

# Auto-start on import (optional - uncomment to enable)
# def auto_init():
#     """Automatically start initialization on import"""
#     init = get_preemptive_init()
#     init.start_background_init()
    
# auto_init()