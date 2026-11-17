"""Interface Streamlit : question texte/image, sources et agent MCP."""
import asyncio
import tempfile
from pathlib import Path
import streamlit as st
from settings import ROOT

st.set_page_config(page_title='RAG multimodal', page_icon='📚', layout='wide')
st.title('RAG multimodal documentaire')
st.write('Interrogez Attention Is All You Need et retrouvez les passages et figures utilisés.')
mode = st.sidebar.radio('Mode', ['RAG direct', 'Agent LangChain', 'Agent via MCP'])
type_resultat = st.sidebar.selectbox('Sources recherchées', ['tous', 'texte', 'image'])
k = st.sidebar.slider('Nombre de sources', 1, 12, 6)
question = st.text_area('Votre question', 'Pourquoi le Transformer utilise-t-il plusieurs têtes d’attention ?')
image = st.file_uploader('Image de recherche (mode RAG direct)', type=['png', 'jpg', 'jpeg', 'webp'])
if mode != 'RAG direct' and image is not None:
    st.info('Choisissez RAG direct pour utiliser cette image comme requête.')

if st.button('Rechercher', type='primary'):
    if not question.strip():
        st.warning('Saisissez une question.')
    elif mode != 'RAG direct' and image is not None:
        st.warning('Ce mode attend une question textuelle. Retirez l’image ou choisissez RAG direct.')
    else:
        try:
            with st.spinner('Recherche documentaire en cours…'):
                if mode == 'RAG direct':
                    from rag_core import interroger, image_source
                    with tempfile.TemporaryDirectory(prefix='rag-requete-') as dossier:
                        chemin = None
                        if image is not None:
                            chemin = Path(dossier) / ('requete' + Path(image.name).suffix.lower())
                            chemin.write_bytes(image.getvalue())
                        resultat = interroger(question, type_resultat, k, image=chemin)
                    st.markdown(resultat['reponse'])
                    st.dataframe(resultat['sources_transmises'], width='stretch')
                    for source in resultat['sources_transmises']:
                        if source['type'] == 'image':
                            chemin = image_source(source['id'])
                            if chemin:
                                st.image(str(chemin), caption=f"{source['pdf']}, page {source['page']}")
                    for message in resultat['avertissements']:
                        st.warning(message)
                else:
                    demande = f'{question}\nTypes de sources : {type_resultat}. Nombre de sources : {k}.'
                    if mode == 'Agent via MCP':
                        from agent_mcp import interroger_agent_mcp
                        resultat = asyncio.run(interroger_agent_mcp(demande))
                    else:
                        from agent_local import creer_agent
                        resultat = creer_agent().invoke({'messages': [{'role': 'user', 'content': demande}]},
                                                       config={'recursion_limit': 20})
                    st.markdown(resultat['messages'][-1].content)
                    with st.expander('Appels documentaires et sources'):
                        for message in resultat['messages']:
                            if message.type == 'tool':
                                st.write(message.content)
        except Exception as erreur:
            st.error(f'La requête a échoué ({type(erreur).__name__}). Vérifiez les fichiers du TP et le fournisseur LLM.')
            # Pas de détail brut pouvant contenir une clé, une URL privée ou un chemin personnel.
