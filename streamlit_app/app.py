import streamlit as st
import os
import json
import time
from dotenv import load_dotenv
from pyairtable import Api as AirtableApi
import anthropic
import pickle
import numpy as np
from openai import OpenAI
from datetime import datetime
from typing import List, Dict, Any, Optional

# Set page configuration
st.set_page_config(
    page_title="Airtable Search with AI",
    page_icon="🔍",
    layout="wide"
)

# Path to the .env file (one directory up)
env_path = os.path.join(os.path.dirname(__file__), '..', 'backend', '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)

# Get credentials from environment variables or Streamlit secrets
try:
    # Check if we're running in Streamlit Cloud with secrets
    _ = st.secrets["ANTHROPIC_API_KEY"]
    # If we get here, secrets exist, so use them
    ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
    AIRTABLE_API_KEY = st.secrets["AIRTABLE_API_KEY"]
    AIRTABLE_BASE_ID = st.secrets["AIRTABLE_BASE_ID"]
    AIRTABLE_TABLE_NAME = st.secrets["AIRTABLE_TABLE_NAME"]
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    st.success("Using Streamlit Cloud secrets!")
except (KeyError, FileNotFoundError):
    # We're in local development, use environment variables
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
    AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
    AIRTABLE_TABLE_NAME = os.getenv("AIRTABLE_TABLE_NAME")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    st.info("Using local environment variables from .env file.")

# Check API keys
if not all([ANTHROPIC_API_KEY, AIRTABLE_API_KEY, AIRTABLE_BASE_ID, AIRTABLE_TABLE_NAME]):
    st.error("One or more required API keys are missing. Please check your .env file.")
    st.stop()

# Initialize Airtable API
airtable_api = AirtableApi(AIRTABLE_API_KEY)
airtable_table = airtable_api.table(AIRTABLE_BASE_ID, AIRTABLE_TABLE_NAME)

# Initialize Anthropic client
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Vector Search class (simplified from our previous implementation)
class VectorSearch:
    def __init__(self, openai_api_key: str, collection_name: str = "airtable_records"):
        """Initialize the vector search with OpenAI API key."""
        # Setup OpenAI client for embeddings
        self.openai_client = OpenAI(api_key=openai_api_key)
        
        # Setup in-memory storage
        self.collection_name = collection_name
        self.embeddings = []
        self.documents = []
        self.metadatas = []
        self.ids = []
        
        # File to save embeddings to
        os.makedirs("vector_db", exist_ok=True)
        self.storage_file = f"vector_db/{collection_name}.pkl"
        
        # Load existing embeddings if available
        self._load_from_disk()
        
        print(f"Vector search initialized with collection '{collection_name}' ({len(self.ids)} records)")
    
    def _create_embedding(self, text: str) -> List[float]:
        """Create an embedding vector for a text using OpenAI API."""
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
            dimensions=1536
        )
        return response.data[0].embedding
    
    def _format_record_text(self, record: Dict[str, Any]) -> str:
        """Format an Airtable record as a single text string for embedding."""
        record_text = f"Record ID: {record['id']}\n"
        
        # Add all fields to the text
        for field_name, value in record['fields'].items():
            # Convert any complex values to strings
            value_str = str(value)
            record_text += f"{field_name}: {value_str}\n"
            
        return record_text
    
    def _save_to_disk(self):
        """Save embeddings to disk."""
        data = {
            "embeddings": self.embeddings,
            "documents": self.documents,
            "metadatas": self.metadatas,
            "ids": self.ids,
            "timestamp": datetime.now().isoformat()
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.storage_file), exist_ok=True)
        
        with open(self.storage_file, "wb") as f:
            pickle.dump(data, f)
        
        print(f"Saved {len(self.ids)} embeddings to {self.storage_file}")
    
    def _load_from_disk(self):
        """Load embeddings from disk if available."""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, "rb") as f:
                    data = pickle.load(f)
                
                self.embeddings = data.get("embeddings", [])
                self.documents = data.get("documents", [])
                self.metadatas = data.get("metadatas", [])
                self.ids = data.get("ids", [])
                
                print(f"Loaded {len(self.ids)} embeddings from {self.storage_file}")
                
            except Exception as e:
                print(f"Error loading embeddings from disk: {e}")
                # Initialize empty data structures
                self.embeddings = []
                self.documents = []
                self.metadatas = []
                self.ids = []
    
    def _cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def add_or_update_records(self, records: List[Dict[str, Any]]) -> None:
        """Add or update records in the vector database."""
        if not records:
            print("No records to add or update")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        print(f"Processing {len(records)} records for vector storage...")
        for i, record in enumerate(records):
            status_text.text(f"Processing record {i+1}/{len(records)}...")
            progress_bar.progress((i+1)/len(records))
            
            # Create a unique ID for the record
            record_id = record['id']
            
            # Format the record as text
            record_text = self._format_record_text(record)
            
            try:
                # Generate embedding for the record
                embedding = self._create_embedding(record_text)
                
                # Store metadata for the record (project, name, etc.)
                metadata = {
                    "record_id": record_id,
                }
                
                # Add important fields to metadata for filtering if they exist
                for key in ["Project", "Name", "Type", "Title"]:
                    if key in record['fields']:
                        metadata[key.lower()] = str(record['fields'][key])
                
                # Check if record already exists
                if record_id in self.ids:
                    # Update existing record
                    idx = self.ids.index(record_id)
                    self.embeddings[idx] = embedding
                    self.documents[idx] = record_text
                    self.metadatas[idx] = metadata
                else:
                    # Add new record
                    self.ids.append(record_id)
                    self.embeddings.append(embedding)
                    self.documents.append(record_text)
                    self.metadatas.append(metadata)
                
            except Exception as e:
                print(f"Error processing record {record_id}: {e}")
            
            # Sleep a bit to avoid rate limits
            time.sleep(0.1)
        
        progress_bar.empty()
        status_text.empty()
        
        # Save to disk
        self._save_to_disk()
        
        print(f"Added/updated {len(records)} records in vector database")
    
    def search(self, query: str, project: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for records similar to the query.
        Optionally filter by project.
        """
        if not self.embeddings:
            return []
        
        # Create query embedding
        query_embedding = self._create_embedding(query)
        
        # Calculate similarity scores
        scores = []
        for i, record_embedding in enumerate(self.embeddings):
            # Skip if we're filtering by project and this record doesn't match
            if project and project != "all":
                metadata = self.metadatas[i]
                if "project" not in metadata or metadata["project"] != project:
                    continue
            
            # Calculate similarity
            score = self._cosine_similarity(query_embedding, record_embedding)
            scores.append((i, score))
        
        # Sort by score (highest first)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get top results
        results = []
        for i, score in scores[:limit]:
            result = {
                "id": self.ids[i],
                "score": float(score),
                "metadata": self.metadatas[i],
                "content": self.documents[i]
            }
            results.append(result)
        
        return results
    
    def total_records(self) -> int:
        """Get the total number of records in the vector database."""
        return len(self.ids)
    
    def clear_all(self) -> None:
        """Clear all records from the vector database."""
        self.embeddings = []
        self.documents = []
        self.metadatas = []
        self.ids = []
        
        # Remove file if it exists
        if os.path.exists(self.storage_file):
            os.remove(self.storage_file)
            
        print(f"Cleared all records from collection '{self.collection_name}'")


# Initialize Vector Search if OpenAI API key is provided
vector_search = None
if OPENAI_API_KEY:
    try:
        vector_search = VectorSearch(OPENAI_API_KEY)
        print("Vector search initialized successfully.")
    except Exception as e:
        st.error(f"Error initializing vector search: {e}")
        print(f"Error initializing vector search: {e}")
else:
    st.warning("OpenAI API key not provided, vector search not available.")
    print("OpenAI API key not provided, vector search not available.")


# Streamlit UI
st.title("🔍 Airtable Search with AI")

# Sidebar for settings and tools
with st.sidebar:
    st.header("Settings & Tools")
    
    # Vector search info
    st.subheader("Vector Search Status")
    if vector_search:
        st.success(f"✅ Indexed records: {vector_search.total_records()}")
    else:
        st.error("❌ Vector search not available")
    
    # Admin section
    st.subheader("Admin Tools")
    
    if st.button("Index All Records", type="primary", disabled=not vector_search):
        with st.spinner("Fetching all records from Airtable..."):
            try:
                all_records = airtable_table.all()
                if not all_records:
                    st.error("No records found in Airtable table.")
                else:
                    st.info(f"Found {len(all_records)} records in Airtable. Starting indexing...")
                    vector_search.add_or_update_records(all_records)
                    st.success(f"✅ Successfully indexed {len(all_records)} records!")
            except Exception as e:
                st.error(f"Error indexing records: {e}")

# Get all available projects
@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_projects():
    try:
        all_records = airtable_table.all()
        if not all_records:
            return []
        
        # Extract project names from records
        project_field = 'Project'
        projects = []
        for record in all_records:
            if project_field in record['fields'] and record['fields'][project_field]:
                # If the project field contains a list, extend the projects list
                if isinstance(record['fields'][project_field], list):
                    projects.extend(record['fields'][project_field])
                # If it's a single value, append it
                else:
                    projects.append(record['fields'][project_field])
        
        # Get unique projects and sort alphabetically
        unique_projects = sorted(set(projects))
        return ["All Projects"] + unique_projects
    except Exception as e:
        st.error(f"Error fetching projects: {e}")
        return ["All Projects"]

# Get projects
projects = get_projects()
selected_project = st.selectbox("Filter by Project:", projects)

# Search interface
query = st.text_input("Enter your query:", placeholder="What would you like to search for?")
search_button = st.button("Search", type="primary")

if search_button and query:
    # Normalize selected project
    if selected_project == "All Projects":
        selected_project = "all"
    
    with st.spinner("Searching..."):
        if vector_search and vector_search.total_records() > 0:
            st.info("Using vector search for more accurate results...")
            # Get search results using vector search
            search_results = vector_search.search(
                query=query,
                project=selected_project if selected_project != "all" else None,
                limit=10
            )
            
            if not search_results:
                st.warning("No relevant information was found for your query.")
            else:
                # Extract the content from search results for Claude to summarize
                contents = [result["content"] for result in search_results]
                relevant_data = "\n\n".join(contents)
                
                # Construct the prompt for Claude to summarize the results
                prompt = (
                    f"You are analyzing search results from an Airtable database.\n\n"
                    f"Here are the most relevant records found for the query: '{query}':\n\n"
                    f"{relevant_data}\n\n"
                    f"Based ONLY on the data provided above, answer this question: '{query}'\n\n"
                    f"If you find relevant information, provide it clearly. If you don't find any relevant information, "
                    f"respond with 'No relevant information found in the database.' Do not make up information or refer to data "
                    f"outside these search results. Cite record IDs when referencing specific information."
                )
                
                # Call Claude to synthesize the results
                with st.spinner("Generating response with AI..."):
                    response = anthropic_client.messages.create(
                        model="claude-3-7-sonnet-latest",
                        max_tokens=1500,
                        messages=[{"role": "user", "content": prompt}]
                    )
                
                # Extract Claude's response
                if response.content and len(response.content) > 0:
                    final_result = response.content[0].text
                else:
                    final_result = "Could not generate a response from the search results."
                
                # Display the results
                st.markdown("### Results")
                st.markdown(final_result)
                
                # Display stats
                st.caption(f"Searched {vector_search.total_records()} records using vector search and found {len(search_results)} relevant results.")
                
                # Advanced: Show raw results in an expander
                with st.expander("View Raw Search Results"):
                    for i, result in enumerate(search_results):
                        st.markdown(f"**Result {i+1}** (Score: {result['score']:.4f})")
                        st.text(result["content"])
                        st.divider()
        else:
            st.error("Vector search is not available or no records have been indexed. Please index your records first.")

# Footer
st.divider()
st.caption("Powered by Anthropic Claude & OpenAI Embeddings") 