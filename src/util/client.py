"""
This implementation of an MCP client is largely taken from the tutorial file from the
official MCP documentation "Build an MCP client"
https://modelcontextprotocol.io/docs/develop/build-client
raw file available here:
https://github.com/modelcontextprotocol/quickstart-resources/blob/main/mcp-client-python/client.py
"""

from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from pathlib import Path
from src.util.utils import get_root_dir

from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env


class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_server(self, command: str, server_args: list | str | Path):
        """Connect to an MCP server"""
        if type(server_args) != list:
            server_args = [server_args]
        if type(server_args) != list:
            raise ValueError('server_args must be a list.')
        # Convert all args to strings in case they aren't already
        server_args = [str(arg) for arg in server_args]

        server_params = StdioServerParameters(
            command=command,
            args=server_args,
            env=None,
            cwd=str(get_root_dir())  # explicitly set CWD to project root to avoid import errors
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools and resources
        tool_response = await self.session.list_tools()
        tools = tool_response.tools
        resource_response = await self.session.list_resources()
        resources = resource_response.resources
        print("\nConnected to server with tools:", [tool.name for tool in tools])
        print("And resources:", [res.name for res in resources])

    async def list_tools(self):
        """List available tools from the connected MCP server"""
        if not self.session:
            raise RuntimeError("Not connected to any MCP server")

        response = await self.session.list_tools()
        return response.tools

    async def list_resources(self):
        """List available resources from the connected MCP server"""
        if not self.session:
            raise RuntimeError("Not connected to any MCP server")

        response = await self.session.list_resources()
        return response.resources

    async def call_tool(self, tool_name: str, tool_args: dict):
        """Execute tool call"""
        result = await self.session.call_tool(tool_name, tool_args)
        return result

    async def cleanup(self):
        """Clean up resources"""
        if self.exit_stack is None:
            return
        try:
            await self.exit_stack.aclose()
        except Exception as e:
            print(f"Warning: Error during client cleanup: {e}")
            # Don't try to close again, just mark as closed
            pass
        finally:
            self.exit_stack = None
            self.session = None
