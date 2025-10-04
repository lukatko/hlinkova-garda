#!/usr/bin/env python3
"""
Test script for the AI agent to verify implementation
"""

import asyncio
from src.agent import Agent

async def test_currency_conversion():
    """Test currency conversion MCP server"""
    print("Testing currency conversion...")
    agent = Agent()
    
    try:
        await agent.initialise_servers()
        
        # Test a simple currency conversion question
        test_question = "Convert 100 USD to EUR using the currency rates"
        
        if agent.currency_client:
            print(f"Testing question: {test_question}")
            answer = await agent.answer_question(test_question)
            print(f"Answer: {answer}")
        else:
            print("Currency client not available")
            
    except Exception as e:
        print(f"Error during test: {e}")
    finally:
        # Cleanup
        if hasattr(agent, 'currency_client') and agent.currency_client:
            await agent.currency_client.cleanup()

async def test_basic_functionality():
    """Test basic agent functionality with a simple question"""
    print("\nTesting basic functionality...")
    agent = Agent()
    
    try:
        await agent.initialise_servers()
        
        # Test with first question from the dataset
        test_question = "What is 2 + 2?"
        
        print(f"Testing simple question: {test_question}")
        answer = await agent.answer_question(test_question)
        print(f"Answer: {answer}")
            
    except Exception as e:
        print(f"Error during test: {e}")
    finally:
        # Cleanup all clients
        for client_name in ['wikipedia_client', 'database_client', 'currency_client']:
            if hasattr(agent, client_name):
                client = getattr(agent, client_name)
                if client:
                    try:
                        await client.cleanup()
                    except:
                        pass

if __name__ == "__main__":
    print("Running AI Agent Tests...")
    
    asyncio.run(test_currency_conversion())
    asyncio.run(test_basic_functionality())
    
    print("\nTest completed!")