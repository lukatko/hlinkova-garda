# AI Agent for Hackathon Question Answering

This project implements an AI agent that answers questions using multiple data sources through Model Context Protocol (MCP) servers.

## Data Sources

The agent has access to three main data sources:

1. **Wikipedia** - For general knowledge and current events
2. **SQL Database (Our World in Data)** - For CO2, energy, and emissions data
3. **PDF Documents** - Annual reports from companies (Erste Group, GSK, Swisscom)
4. **Currency Conversion** - Exchange rates relative to EUR

## Architecture

### MCP Servers
- `src/mcp_servers/database.py` - Connects to remote SQL Server database
- `src/mcp_servers/currency_converter.py` - Handles currency conversions
- Wikipedia MCP server (external dependency)

### Main Components
- `src/agent.py` - Main AI agent that orchestrates tool calls
- `src/util/client.py` - MCP client implementation
- `src/util/utils.py` - Utility functions for path management

## Setup Instructions

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt  # If requirements.txt exists
   # Or manually install: anthropic, mcp, python-dotenv, pymssql
   ```

2. **Environment Variables**
   Create a `.env` file with:
   ```
   ANTHROPIC_API_KEY=your_claude_api_key_here
   SQLSERVER_HOST=your_database_host
   SQLSERVER_PORT=1433
   SQLSERVER_USER=your_db_user
   SQLSERVER_PASSWORD=your_db_password
   SQLSERVER_DB=your_database_name
   ```

3. **Run the Agent**
   ```bash
   python src/agent.py
   ```

4. **Test the Implementation**
   ```bash
   python test_agent.py
   ```

## Key Features

### Question Processing
- Loads questions from `data/public_questions.json`
- Uses Claude 3.5 Haiku for cost-effective processing
- Implements iterative tool calling with max iteration limits

### Tool Integration
- **Wikipedia Tools**: Search and retrieve articles
- **Database Tools**: Query CO2, energy, and emissions data
- **Currency Tools**: Convert between currencies using official rates

### Source Attribution
- Tracks sources used in answers
- Supports Wikipedia, PDF, and database sources
- Maintains page numbers for PDF sources

### Output Format
- Generates `submission.json` in the required format
- Includes team name, questions, answers, and sources
- Compatible with hackathon submission requirements

## Usage Example

```python
import asyncio
from src.agent import Agent

async def run_agent():
    agent = Agent()
    await agent.initialise_servers()
    
    question = "What were the CO2 emissions for Austria in 2000?"
    answer = await agent.answer_question(question)
    print(f"Answer: {answer}")
    
    # Cleanup
    await agent.cleanup()

asyncio.run(run_agent())
```

## Implementation Status

✅ **Completed:**
- Currency converter MCP server implementation
- Question loading from JSON
- Answer generation with tool calling
- Output formatting for submission
- Source tracking structure
- Multi-server coordination

🔄 **Ready for Enhancement:**
- Source extraction from AI responses
- Error handling improvements
- Performance optimizations
- Additional data source integrations

## Files Structure

```
src/
├── agent.py                 # Main AI agent
├── mcp_servers/
│   ├── currency_converter.py # Currency conversion tools
│   └── database.py          # Database query tools
└── util/
    ├── client.py            # MCP client implementation
    └── utils.py             # Utility functions

data/
├── public_questions.json    # Questions to answer
├── sample_submission.json   # Expected output format
├── currencies/
│   ├── currency_rates.json # Exchange rates (EUR base)
│   └── currency_names.json # Currency names
└── annual_reports/          # PDF documents

test_agent.py               # Test script
submission.json             # Generated output
```

## Notes for Hackathon

- Uses Claude 3.5 Haiku for cost efficiency during development/testing
- Can be upgraded to Claude Sonnet or Opus for better performance
- Implements proper cleanup to avoid resource leaks
- Structured for easy debugging and monitoring
- Ready for team name customization in submission output