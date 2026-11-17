"""Vérifier les décisions, observations et bornes du graphe sans LLM externe."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from langchain_core.messages import AIMessage, HumanMessage
from langchain.tools import tool
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_react import creer_agent_react, analyser_trace
from retriever_tool import retrouver


@tool
def source_test(texte: str) -> str:
    """Rechercher une source de test."""
    return '{"sources":[{"pdf":"article.pdf","page":4,"type":"texte"}]}'


class CerveauSimule:
    def __init__(self, messages):
        self.messages = iter(messages)

    def bind_tools(self, outils):
        return self

    def invoke(self, messages):
        return next(self.messages)


class GrapheTests(unittest.TestCase):
    def test_cycle_action_observation_reponse(self):
        cerveau = CerveauSimule([
            AIMessage(content='', tool_calls=[{'id':'appel1','name':'source_test','args':{'texte':'attention'}}]),
            AIMessage(content='Réponse [article.pdf, p. 4]')])
        resultat = creer_agent_react(cerveau, [source_test]).invoke({'messages':[HumanMessage(content='Question')]})
        trace = analyser_trace(resultat)
        self.assertEqual([e['etape'] for e in trace], ['Action','Observation','Réponse finale'])
        self.assertEqual(trace[1]['appel_id'], trace[0]['appel_id'])
        self.assertEqual(trace[1]['resultat']['sources'][0]['page'], 4)

    def test_refuse_conclusion_sans_recherche(self):
        agent = creer_agent_react(CerveauSimule([AIMessage(content='Réponse non documentée')]), [source_test])
        with self.assertRaisesRegex(RuntimeError, 'consulter'):
            agent.invoke({'messages':[HumanMessage(content='Question')]})

    def test_refuse_quatrieme_appel_avant_execution(self):
        appels = [{'id':str(i),'name':'source_test','args':{'texte':'x'}} for i in range(4)]
        agent = creer_agent_react(CerveauSimule([AIMessage(content='',tool_calls=appels)]), [source_test])
        with self.assertRaisesRegex(RuntimeError, 'trois appels'):
            agent.invoke({'messages':[HumanMessage(content='Question')]})

    def test_identifiant_image_ne_peut_pas_traverser_les_dossiers(self):
        with self.assertRaisesRegex(ValueError, 'Identifiant'):
            retrouver(image_id='input:../../.env')

    def test_refuse_deux_modalites_dans_un_meme_appel(self):
        with self.assertRaises(ValueError):
            retrouver(texte='question', image_id='image')


if __name__ == '__main__':
    unittest.main()
