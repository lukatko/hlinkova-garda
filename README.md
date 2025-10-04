# AIM Hackathon - Agentic AI for Sustainability Questions
4th of October 2025

[//]: # (<img src="assets/flyer.png" alt="Hackathon Flyer" width="400"/>)

## Getting Started
1. Clone repository
2. Change ".env.template" to ".env" and add your Anthropic API key
3. Set up the package manager [uv](https://docs.astral.sh/uv/getting-started/installation/): `pip install uv`
4. Set up environment with uv: `uv sync`
5. Add new packages (as with pip): `uv add <package-name>`

## Start Developing (Hints)
1. Start with the agent.py file
1. Start small -> e.g. answer easy SQL questions
2. Evaluate from the beginning
   - submit to the public leaderboard only 1-3 questions to get familiar with the process and format
3. Then go big, one step at a time
   - add easy Wikipedia questions
   - add easy PDF questions
   - then medium, then hard


## Big Picture

```mermaid
flowchart LR
    %% Nodes
    PubQ["Public Question Set (with answers)"]
    PrivQ["Private Question Set (for final eval)"]
    Agent["Agent (challenge)"]
    VecDB["Vector DB"]
    LocPy["Local Python tool"]
    RemDB["Remote SQL DB"]
    Curr["Currency converter"]
    Wiki["Wikipedia MCP server"]
    Ans["Unambiguous answers: float, categories, etc."]
    Claude["Anthropic (Claude)"]

    %% Flows
    PubQ -->|available all of the time| Agent
    PrivQ -.->|released at the end| Agent
    Agent -->|output| Ans
    
    Agent <--> LocPy
    Agent <--> RemDB
    Agent <--> VecDB
    Agent <--> Curr
    Agent <--> Wiki
    Agent <--> Claude
```


## Leaderboard and Submission
- [Main page](https://hackathon-server.ashysand-de33d6c5.westeurope.azurecontainerapps.io/)
- [All API endpoints](https://hackathon-server.ashysand-de33d6c5.westeurope.azurecontainerapps.io/docs)
- [Leaderboard page](https://hackathon-server.ashysand-de33d6c5.westeurope.azurecontainerapps.io/leaderboard)
  - showing the best submission of each team only
- To check validity of your submission json, use the validation endpoint `/check-submission`
- To submit your results during the event, use the submission endpoint `/submit`
- To submit your **final results of the private set**, use the submission endpoint `/submit-private`

### Automatic submission script

There's a small helper script to upload a submission JSON to the server:

   scripts/submit_submission.py

It uses `requests` and supports a dry-run. Example:

```bash
python scripts/submit_submission.py --file data/sample_submission.json --endpoint submit
```

If `requests` is not installed you can add it using the project's `uv` workflow or pip:

```bash
uv add requests
# or
pip install requests
```



## Nice reads
* [Develop with Claude API](https://docs.claude.com/en/docs/get-started#python)
* [Claude JSON Mode](https://docs.claude.com/en/docs/test-and-evaluate/strengthen-guardrails/increase-consistency)
* [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)
* [Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk)
* [Model Context Protocol (MCP)](https://modelcontextprotocol.io/docs/develop/build-server)
* [Claude Embeddings](https://docs.claude.com/en/docs/build-with-claude/embeddings)
