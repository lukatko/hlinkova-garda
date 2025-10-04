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

from anthropic import Anthropic
from dotenv import load_dotenv
import os
import json

load_dotenv()  # load environment variables from .env

class Agent:
    def __init__(self):
        self.tools = []
        self.anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")) # Initialise Claude

    async def initialise_servers(self):

        # Initialize tools lists
        self.wikipedia_tools = []
        self.database_tools = []
        self.currency_tools = []

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

        self.tools = self.wikipedia_tools + self.database_tools + self.currency_tools

    async def answer_question(self, question: str) -> str:
        """
        Answer a question by calling tools as needed.
        :param question: a single question as a string
        :return: the answer as a string
        """
        messages = [
            {
                "role": "user",
                "content": f"""Answer this question using the available tools: {question}

You have access to:
- Wikipedia tools for general knowledge and current events
- Database tools for querying CO2, energy, and emissions data from Our World in Data
- Currency conversion tools for converting between different currencies

Important guidelines:
1. Always cite your sources in your response
2. Use specific tool calls to gather information
3. For numerical data, provide precise values when possible
4. If you need to convert currencies, use the currency conversion tool
5. If a question requires calculations, show your work
6. If information is not available, clearly state that"""
            }
        ]

        max_iterations = 10  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Get a response from Claude
            response = self.anthropic.messages.create(
                model="claude-3-5-haiku-20241022",  # Using fastest/cheapest model for hackathon
                max_tokens=4000,
                messages=messages,
                tools=self.tools
            )

            # Add Claude's response to the conversation
            messages.append({
                "role": "assistant", 
                "content": response.content
            })

            # Check if Claude wants to use tools
            if response.stop_reason == "tool_use":
                tool_results = []
                
                for content_block in response.content:
                    if content_block.type == "tool_use":
                        tool_name = content_block.name
                        tool_input = content_block.input
                        tool_call_id = content_block.id
                        
                        try:
                            # Call the appropriate tool based on which client handles it
                            result = await self._call_tool(tool_name, tool_input)
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_call_id,
                                "content": str(result)
                            })
                        except Exception as e:
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
            else:
                # No more tool calls, extract the final answer
                final_response = ""
                for content_block in response.content:
                    if content_block.type == "text":
                        final_response += content_block.text
                return final_response.strip()

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
        
        raise Exception(f"Tool {tool_name} not found in any available clients")


async def main(verbose: bool = True):
    agent = Agent()
    
    # Load questions from public_questions.json
    questions_file = get_root_dir() / 'data' / 'public_questions.json'
    with open(questions_file, 'r') as f:
        questions_data = json.load(f)
    
    # Extract just the question text from the data structure
    questions = [questions_data[str(i)]["question"] for i in range(1, len(questions_data) + 1)]
    
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
    answers = asyncio.run(main())
