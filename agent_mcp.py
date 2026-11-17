"""L'agent ne charge pas le RAG : il découvre ses outils par MCP."""
import argparse
import asyncio
import sys
from pathlib import Path
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from settings import ROOT, llm_config, llm_options
from agent_local import INSTRUCTIONS


def client_mcp():
    return MultiServerMCPClient({'rag': {
        'transport': 'stdio', 'command': sys.executable,
        'args': [str(ROOT / 'mcp_server.py')],
    }})


async def interroger_agent_mcp(question):
    # Une session conserve le serveur et le modèle pendant toute l'invocation.
    from langchain_mcp_adapters.tools import load_mcp_tools
    client = client_mcp()
    async with client.session('rag') as session:
        tools = await load_mcp_tools(session)
        agent = create_agent(
            model=ChatOpenAI(**llm_config(), **llm_options(), temperature=0, timeout=180, max_retries=1, max_tokens=1500),
            tools=tools, system_prompt=INSTRUCTIONS,
            middleware=[ToolCallLimitMiddleware(tool_name='consulter_pdf', run_limit=3, exit_behavior='error')])
        return await agent.ainvoke({'messages': [{'role': 'user', 'content': question}]},
                                   config={'recursion_limit': 20})


async def verifier_mcp():
    tools = await client_mcp().get_tools()
    assert {'consulter_pdf', 'retriever_tool'} <= {t.name for t in tools}
    print('Connexion MCP réussie :', ', '.join(t.name for t in tools))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Agent documentaire utilisant un serveur MCP séparé.')
    parser.add_argument('question', nargs='?', default='Pourquoi utiliser plusieurs têtes d’attention ?')
    parser.add_argument('--check', action='store_true', help='Vérifier la découverte MCP sans charger le LLM.')
    args = parser.parse_args()
    if args.check:
        asyncio.run(verifier_mcp())
    else:
        result = asyncio.run(interroger_agent_mcp(args.question))
        print(result['messages'][-1].content)
