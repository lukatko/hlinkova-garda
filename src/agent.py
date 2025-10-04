"""
This script acts as the MCP host, the AI agent which calls tools to answer questions.

Most of the script was copied and adapted from the official MCP documentation "Build an MCP client" available at:
https://modelcontextprotocol.io/docs/develop/build-client
raw file available here:
https://github.com/modelcontextprotocol/quickstart-resources/blob/main/mcp-client-python/client.py
"""

import asyncio
from src.util.client import MCPClient
from src.util.utils import get_root_dir
from src.mcp_servers.database import get_tables, get_schema

from anthropic import Anthropic
from dotenv import load_dotenv
import os
import json

load_dotenv()  # load environment variables from .env

class Agent:
    def __init__(self):
        self.tools = []
        self.anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")) # Initialise Claude
        
        # Add built-in math tool
        self.math_tools = [{
            "name": "calculate",
            "description": "Execute Python mathematical expressions safely. Use for arithmetic, percentages, conversions, etc.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Python mathematical expression to evaluate (e.g., '(100 * 1.1672)', '(67949000 - 22427000) / 22427000 * 100')"
                    }
                },
                "required": ["expression"]
            }
        }]
        
        # Add built-in vector search tool
        self.vector_search_tools = [{
            "name": "search_pdf_documents", 
            "description": "Search through annual reports and sustainability documents using semantic similarity to find relevant information about companies, emissions, sustainability metrics, safety data, accidents, etc. Searches across all PDF content including Erste Group, GSK, and Swisscom reports.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to find relevant information in PDF documents (e.g., 'Swisscom carbon emissions', 'Erste Group accidents', 'GSK environmental impact', 'health safety metrics')"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 8, max: 15)",
                        "default": 8
                    }
                },
                "required": ["query"]
            }
        }]
    
    def calculate(self, expression: str) -> dict:
        """
        Safely execute Python mathematical expressions.
        """
        try:
            # Only allow safe mathematical operations
            allowed_names = {
                "__builtins__": {},
                "abs": abs, "round": round, "min": min, "max": max,
                "sum": sum, "pow": pow, "divmod": divmod,
                # Math functions
                "math": __import__("math"),
            }
            
            print(f"DEBUG: Calculating expression: {expression}")
            result = eval(expression, allowed_names, {})
            print(f"DEBUG: Calculation result: {result}")
            
            return {
                "expression": expression,
                "result": result,
                "type": type(result).__name__
            }
        except Exception as e:
            print(f"DEBUG: Calculation error: {e}")
            return {
                "expression": expression,
                "error": str(e),
                "result": None
            }

    def search_pdf_documents(self, query: str, top_k: int = 5) -> dict:
        """
        Search through PDF documents using Azure OpenAI embeddings and ChromaDB.
        """
        try:
            import chromadb
            from openai import OpenAI
            
            # Initialize Azure OpenAI client
            endpoint = "https://aim-azure-ai-foundry.cognitiveservices.azure.com/openai/v1/"
            deployment_name = "text-embedding-model"
            api_key = os.getenv("AZURE_API_KEY")
            
            openai_client = OpenAI(
                base_url=endpoint,
                api_key=api_key,
            )
            
            # Initialize ChromaDB
            client = chromadb.PersistentClient(path="./chroma_db")
            collection = client.get_collection("pdf_chunks")
            
            print(f"DEBUG: Searching PDF documents for: {query}")
            
            # Limit top_k but increase for better coverage
            top_k = min(top_k, 15)
            search_k = min(top_k * 3, 50)
            
            # Generate embedding using Azure OpenAI
            response = openai_client.embeddings.create(
                input=query,
                model=deployment_name
            )
            query_embedding = response.data[0].embedding
            
            # Search in ChromaDB
            results_data = collection.query(
                query_embeddings=[query_embedding],
                n_results=search_k
            )
            
            # Second: Get all documents for text-based matching
            all_docs_data = collection.get()
            
            # Prepare results containers
            results = []

            # Process semantic search results
            if results_data['documents'] and results_data['documents'][0]:
                for i in range(len(results_data['documents'][0])):
                    metadata = results_data['metadatas'][0][i] if results_data['metadatas'] else {}
                    distance = results_data['distances'][0][i] if results_data['distances'] else 0.0
                    
                    results.append({
                        "document": metadata.get("doc", "Unknown"),
                        "page": metadata.get("page", "Unknown"), 
                        "text": results_data['documents'][0][i],
                        "similarity_score": float(distance),
                        "rank": i + 1
                    })
            
            # print(f"DEBUG: Found {len(results)} PDF document results")
            return {
                "query": query,
                "results": results,
                "total_found": len(results)
            }
            
        except Exception as e:
            print(f"DEBUG: PDF search error: {e}")
            return {
                "query": query,
                "error": str(e),
                "results": []
            }

    async def initialise_servers(self):

        # Initialize tools lists
        self.wikipedia_tools = []
        self.database_tools = []
        self.currency_tools = []
        
        self.vector_db_tools = []

        # Initialise the Wikipedia MCP server
        try:
            self.wikipedia_client = MCPClient()
            await self.wikipedia_client.connect_to_server("python", ["-m", "wikipedia_mcp", "--transport", "stdio"])

            wikipedia_tools = await self.wikipedia_client.list_tools()
            self.wikipedia_tools = [{
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            } for tool in wikipedia_tools]
            self.wikipedia_tools_info = "\n".join(
    [f"- {t['name']}: {t['description']}, expects input: {t['input_schema']}" for t in self.wikipedia_tools]
)
            print("Wikipedia MCP server initialized successfully")
        except Exception as e:
            print(f"Warning: Failed to initialize Wikipedia MCP server: {e}")
            # Ensure client is properly cleaned up if initialization fails
            if hasattr(self, 'wikipedia_client') and self.wikipedia_client is not None:
                try:
                    await self.wikipedia_client.cleanup()
                except:
                    pass
            self.wikipedia_client = None

        # Initialise the OWID database MCP server
        try:
            self.database_client = MCPClient()
            await self.database_client.connect_to_server("python", ["-m", "src.mcp_servers.database"])

            database_tools = await self.database_client.list_tools()
            self.database_tools = [{
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            } for tool in database_tools]
            print("Database MCP server initialized successfully")
        except Exception as e:
            print(f"Warning: Failed to initialize Database MCP server: {e}")
            # Ensure client is properly cleaned up if initialization fails
            if hasattr(self, 'database_client') and self.database_client is not None:
                try:
                    await self.database_client.cleanup()
                except:
                    pass
            self.database_client = None

        # Initialise the Currency Converter MCP server
        try:
            self.currency_client = MCPClient()
            await self.currency_client.connect_to_server("python", ["-m", "src.mcp_servers.currency_converter"])

            currency_tools = await self.currency_client.list_tools()
            self.currency_tools = [{
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            } for tool in currency_tools]
            print("Currency Converter MCP server initialized successfully")
        except Exception as e:
            print(f"Warning: Failed to initialize Currency Converter MCP server: {e}")
            # Ensure client is properly cleaned up if initialization fails
            if hasattr(self, 'currency_client') and self.currency_client is not None:
                try:
                    await self.currency_client.cleanup()
                except:
                    pass
            self.currency_client = None

        
        
        # Initialise the Vector Database MCP server
        try:
            self.vector_db_client = MCPClient()
            await self.vector_db_client.connect_to_server("python", ["-m", "src.mcp_servers.vector_db"])

            vector_db_tools = await self.vector_db_client.list_tools()
            self.vector_db_tools = [{
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            } for tool in vector_db_tools]
            print("Vector Database MCP server initialized successfully")
        except Exception as e:
            print(f"Warning: Failed to initialize Vector Database MCP server: {e}")
            # Ensure client is properly cleaned up if initialization fails
            if hasattr(self, 'vector_db_client') and self.vector_db_client is not None:
                try:
                    await self.vector_db_client.cleanup()
                except:
                    pass
            self.vector_db_client = None

        self.tools = self.wikipedia_tools + self.database_tools + self.currency_tools + self.math_tools + self.vector_db_tools + self.vector_search_tools

    async def answer_question(self, question: str) -> str:
        """
        Answer a question by calling tools as needed.
        :param question: a single question as a string
        :return: the answer as a string
        """
        print(f"DEBUG: Starting to answer question: {question}")
        

        messages = [
            {
                "role": "user",
                "content": f"""Answer this question using the available tools: {question['question']},


You have access to:
<<<<<<< HEAD
- Wikipedia tools for general knowledge and current events:
{self.wikipedia_tools_info}
- Database tools for querying CO2, energy, and emissions data from Our World in Data
=======
- Wikipedia tools for general knowledge and current events
- Database tools for querying CO2, energy, and emissions data from Our World in Data - Please use the resource to check the available tables and schemas
>>>>>>> bab875217b3a6ba772d0b5cae5fbbd08098e3023
- Currency conversion tools for converting between different currencies
- Calculate tool for mathematical operations (addition, subtraction, multiplication, division, percentages, etc.)

If you see 2025 in the question look in the wikipedia first.

ANSWER FORMAT REQUIREMENTS:
Your answer must be in the EXACT format shown below with data type {question['answer_type']} in units {question['unit']}. Do not include explanations, or additional text.
Just provide the raw answer value that matches the expected data type.

Expected answer format examples:
- For numbers: 42 or 42.5 or 412880.659 (not "42" or "42 million")
- For booleans: true or false (not "yes" or "no")  
- For strings: "Potomac River" (include quotes for string answers)
- For null answers: null (when information is not available)

<<<<<<< HEAD
Important workflow for Wikipedia:
1. Start with the **Wikipedia search tool** to find candidate pages.
2. Use the **Wikipedia page content tool** to extract information.
3. If Wikipedia only contains **partial information**:
   - Supplement it with **Database tools** (for numerical values like emissions, GDP, etc.).
   - Use **Currency tools** when conversions are needed.
   - Use your own reasoning to combine results.
4. Always **combine sources** into a single, clear answer.
5. Explicitly list **all sources you used** in the final JSON:
   - Wikipedia → `"source_type": "wikipedia"`, `"source_name": "<page title>"`.
   - Database → `"source_type": "database"`, `"source_name": "owid_co2_data"`.
   - Currency → `"source_type": "internal"`, `"source_name": "currency_rates.json"`.
   - If a tool cannot find data, clearly say so instead of guessing.


=======
>>>>>>> bab875217b3a6ba772d0b5cae5fbbd08098e3023
- If multiple tools were used, include all sources in the list.
- If no sources are available, set `"sources": null`.

Guidelines:
- Provide precise values when possible.
- Show calculations if you derived a result.
- Do not hallucinate data (e.g., no "Scope 5 emissions").
- If information is missing in all tools, clearly state that.

Sources format (when available):
- For PDF files: {{"source_name": "filename.pdf", "source_type": "pdf", "page_number": 123}}
- For Wikipedia: {{"source_name": "Article Title", "source_type": "wikipedia", "page_number": null}}
- For database: {{"source_name": "table_name", "source_type": "database", "page_number": null}}
- When no sources: null

CRITICAL: 
- If the question requires database information, first generate and execute a SQL query using the query_database tool
- Use the exact table and column names shown in the schemas above
- For numerical answers, provide precise values without units or explanations
- If information is not available, return null
- Do NOT include explanatory text in your final answer"""
            }
        ]

        print(f"DEBUG: Initial prompt prepared, total length: {messages[0]} chars")

        max_iterations = 10  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            print(f"DEBUG: Starting iteration {iteration}")
            
            # Get a response from Claude
            print("DEBUG: Calling Claude API...")
            response = self.anthropic.messages.create(
                model="claude-3-5-haiku-20241022",  # Using fastest/cheapest model for hackathon
                max_tokens=8000,
                temperature=0.0,  # Low temperature for more consistent, factual answers
                messages=messages,
                tools=self.tools
            )
            print(f"DEBUG: Claude response received, stop_reason: {response.stop_reason}")

            # Add Claude's response to the conversation
            messages.append({
                "role": "assistant", 
                "content": response.content
            })

            # Check if Claude wants to use tools
            if response.stop_reason == "tool_use":
                print("DEBUG: Claude wants to use tools")
                tool_results = []
                
                for content_block in response.content:
                    if content_block.type == "tool_use":
                        tool_name = content_block.name
                        tool_input = content_block.input
                        tool_call_id = content_block.id
                        
                        print(f"DEBUG: Calling tool '{tool_name}' with input: {tool_input}")
                        
                        try:
                            # Call the appropriate tool based on which client handles it
                            result = await self._call_tool(tool_name, tool_input)
                            print(f"DEBUG: Tool '{tool_name}' result: {str(result)[:200]}...")
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_call_id,
                                "content": str(result)
                            })
                        except Exception as e:
                            print(f"DEBUG: Error calling tool '{tool_name}': {e}")
                            tool_results.append({
                                "type": "tool_result", 
                                "tool_use_id": tool_call_id,
                                "content": f"Error calling tool {tool_name}: {str(e)}"
                            })

                # Add tool results to conversation
                messages.append({
                    "role": "user",
                    "content": tool_results
                })
                print(f"DEBUG: Added {len(tool_results)} tool results to conversation")
            else:
                print("DEBUG: Claude finished, extracting final answer")
                # No more tool calls, extract the final answer
                final_response = ""
                for content_block in response.content:
                    if content_block.type == "text":
                        final_response += content_block.text
                print(f"DEBUG: Final answer length: {len(final_response)} chars")
                return final_response.strip()

        print("DEBUG: Maximum iterations reached!")
        return "Maximum iterations reached. Unable to complete the answer."

    async def _call_tool(self, tool_name: str, tool_input: dict):
        """Helper method to route tool calls to the appropriate MCP client"""
        
        # Check which client handles this tool
        if any(tool["name"] == tool_name for tool in self.wikipedia_tools):
            if self.wikipedia_client:
                return await self.wikipedia_client.call_tool(tool_name, tool_input)
        elif any(tool["name"] == tool_name for tool in self.database_tools):
            if self.database_client:
                return await self.database_client.call_tool(tool_name, tool_input)
        elif any(tool["name"] == tool_name for tool in self.currency_tools):
            if self.currency_client:
                return await self.currency_client.call_tool(tool_name, tool_input)
        elif any(tool["name"] == tool_name for tool in self.math_tools):
            # Handle built-in math tool
            if tool_name == "calculate":
                return self.calculate(tool_input.get("expression", ""))
        elif any(tool["name"] == tool_name for tool in self.vector_search_tools):
            # Handle built-in vector search tool
            if tool_name == "search_pdf_documents":
                return self.search_pdf_documents(
                    tool_input.get("query", ""), 
                    tool_input.get("top_k", 5)
                )
        elif any(tool["name"] == tool_name for tool in self.vector_db_tools):
            if self.vector_db_client:
                return await self.vector_db_client.call_tool(tool_name, tool_input)
        
        raise Exception(f"Tool {tool_name} not found in any available clients")


async def main(verbose: bool = True):
    agent = Agent()
    
    # Load questions from public_questions.json
    questions_file = get_root_dir() / 'data' / 'public_questions.json'
    with open(questions_file, 'r') as f:
        questions_data = json.load(f)
    
    # Extract just the question text from the data structure
    questions = [questions_data[str(i)] for i in range(1, 10)]
    
    answers = []
    try:
        await agent.initialise_servers()
        # TODO: change this loop as needed
        for i, q in enumerate(questions, 1):
            if verbose:
                print(f"Answering question {i}/{len(questions)}...")
            try:
                answer = await agent.answer_question(q)
                answers.append(answer)
                if verbose:
                    print(f"{i}.\nQuestion: {q}\nAnswer: {answer}\n")
            except Exception as e:
                print(f"Error answering question {i}: {e}")
                answers.append(None)  # keep placeholder so indexes match
    finally:
        # Improved cleanup with cancellation handling
        cleanup_tasks = []
        
        if hasattr(agent, 'wikipedia_client') and agent.wikipedia_client is not None:
            try:
                await agent.wikipedia_client.cleanup()
            except Exception as e:
                print(f"Warning: Error cleaning up Wikipedia client: {e}")
            finally:
                agent.wikipedia_client = None

        if hasattr(agent, 'database_client') and agent.database_client is not None:
            try:
                await agent.database_client.cleanup()
            except Exception as e:
                print(f"Warning: Error cleaning up Database client: {e}")
            finally:
                agent.database_client = None
        
        if hasattr(agent, 'currency_client') and agent.currency_client is not None:
            try:
                await agent.currency_client.cleanup()
            except Exception as e:
                print(f"Warning: Error cleaning up Currency client: {e}")
            finally:
                agent.currency_client = None

        if hasattr(agent, 'vector_db_client') and agent.vector_db_client is not None:
            try:
                await asyncio.wait_for(agent.vector_db_client.cleanup(), timeout=5.0)
            except asyncio.TimeoutError:
                print(f"Warning: Vector DB client cleanup timed out")
            except asyncio.CancelledError:
                print(f"Warning: Vector DB client cleanup was cancelled")
            except Exception as e:
                print(f"Warning: Error cleaning up Vector DB client: {e}")
            finally:
                agent.vector_db_client = None
    
    # Save answers in the required JSON format
    output_file = get_root_dir() / 'submission.json'
    submission = {
        "team_name": "hackathon_team",  # TODO: change this to your team name
        "answers": {}
    }
    
    for i, answer in enumerate(answers, 1):
        if answer is not None:
            submission["answers"][str(i)] = {
                "question": questions[i-1],
                "answer": answer,
                "sources": []  # TODO: extract sources from the answer if possible
            }
    
    with open(output_file, 'w') as f:
        json.dump(submission, f, indent=2)
    
    if verbose:
        print(f"Answers saved to {output_file}")
    
    return answers

if __name__ == "__main__":
    try:
        answers = asyncio.run(main())
        print(f"Successfully generated {len([a for a in answers if a is not None])} answers")
    except KeyboardInterrupt:
        print("Agent execution interrupted by user")
    except Exception as e:
        print(f"Agent execution failed: {e}")
        import traceback
        traceback.print_exc()
