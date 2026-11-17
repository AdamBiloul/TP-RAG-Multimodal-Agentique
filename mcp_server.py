"""Expose le RAG via MCP sur stdio ; aucun message applicatif sur stdout."""
import contextlib
import json
import sys
from typing import Literal
from mcp.server.fastmcp import FastMCP

mcp = FastMCP('RAG multimodal documentaire')


@mcp.tool()
def consulter_pdf(question: str, type_resultat: Literal['tous', 'texte', 'image'] = 'tous', k: int = 6) -> str:
    """Rechercher dans les PDF indexés et générer une réponse documentée."""
    # Les sorties de chargement et d'indexation ne doivent pas corrompre stdio.
    with contextlib.redirect_stdout(sys.stderr):
        from rag_core import interroger
        resultat = interroger(question, type_resultat, k)
    return json.dumps(resultat, ensure_ascii=False)


@mcp.tool(name='retriever_tool')
def rechercher_sources(texte: str | None = None, image_id: str | None = None,
                       type_resultat: Literal['tous', 'texte', 'image'] = 'tous', k: int = 6) -> str:
    """Récupérer les sources brutes par texte OU identifiant image, sans génération LLM."""
    with contextlib.redirect_stdout(sys.stderr):
        from retriever_tool import retrouver
        resultat = retrouver(texte, image_id, type_resultat, k)
    return json.dumps(resultat, ensure_ascii=False)


if __name__ == '__main__':
    mcp.run(transport='stdio')
