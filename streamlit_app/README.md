# Airtable Search with AI (Streamlit App)

This Streamlit app lets you search your Airtable database using AI-powered vector search, giving you smart, semantic search results.

## Features

- **Semantic Search**: Uses OpenAI embeddings to find relevant information based on meaning, not just keywords
- **Project Filtering**: Filter search results by project
- **AI-Generated Responses**: Uses Anthropic Claude to synthesize and explain search results
- **Search History**: View the raw results that informed the AI's response
- **Easy Indexing**: One-click indexing of your Airtable data

## Setup Instructions

### Prerequisites

- Python 3.9+
- An OpenAI API key
- An Anthropic API key
- An Airtable API key and base/table information

### Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up API keys:**
   - Copy the `.env` file from the backend directory to include your API keys:
     ```
     ANTHROPIC_API_KEY=your_anthropic_key
     AIRTABLE_API_KEY=your_airtable_key
     AIRTABLE_BASE_ID=your_base_id
     AIRTABLE_TABLE_NAME=your_table_name
     OPENAI_API_KEY=your_openai_key
     ```

3. **Run the app:**
   ```bash
   streamlit run app.py
   ```

4. **First-time setup:**
   - When the app launches, click "Index All Records" in the sidebar to create vector embeddings
   - This process takes a few minutes, depending on the size of your Airtable
   - Once indexed, the vectors are saved to disk for faster startup next time

## Deploying to Streamlit Cloud

1. Push this code to a GitHub repository

2. Go to [Streamlit Sharing](https://share.streamlit.io/) and sign in

3. Click "New app" and select your repository, branch, and app file

4. In the "Advanced settings", add your API keys as secrets:
   ```
   ANTHROPIC_API_KEY = "your_key_here"
   AIRTABLE_API_KEY = "your_key_here"
   AIRTABLE_BASE_ID = "your_base_id_here"
   AIRTABLE_TABLE_NAME = "your_table_name_here"
   OPENAI_API_KEY = "your_key_here"
   ```

5. Deploy the app

6. After the app is deployed, use the "Index All Records" button to create the vector database

## Customization

You can customize this app by:

- Changing the prompt used to interact with Claude
- Adjusting the number of search results displayed
- Adding additional filtering options
- Enhancing the UI with your branding 