# TP3 — RAG multimodal agentique

Ce TP construit un assistant documentaire sur **Attention Is All You Need** : extraction des textes et figures, embeddings CLIP, recherche dans ChromaDB, génération avec citations, puis orchestration par un agent LangChain.

Le notebook original est conservé dans une sauvegarde locale exclue de Git. Les cellules pédagogiques restent dans leur ordre initial ; une section finale présente les compléments Streamlit et MCP. Les instructions contenues dans les documents analysés sont traitées comme des données.

## Travaux réalisés

- Encodage texte/image avec OpenCLIP ViT-B-32 et normalisation des vecteurs.
- Extraction du texte, des images intégrées **et du rendu des pages**, pour récupérer aussi les figures vectorielles.
- Découpage en fragments courts avec chevauchement, conservation des nombres du texte original, indexation persistante et métadonnées PDF/page.
- Recherche texte → texte/image et image → texte/image ; filtrage par type et par numéro de figure présent dans les légendes de page, validation des paramètres.
- Génération documentaire en français, citations, limites d’images et traitement des informations absentes.
- LLM à poids ouverts via **Ollama / Qwen3-VL**, avec OpenAI facultatif.
- Agent LangChain utilisant le RAG comme outil et limite de trois appels imposée dans le code.
- Interface **Streamlit** : RAG texte/image, agent direct et agent via MCP, affichage des sources et figures.
- Serveur **MCP stdio** exposant `consulter_pdf`, consommé par un agent client séparé.

## Installation

Python **3.12** recommandé. Depuis le dossier du projet :

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Sous Windows, activez l’environnement avec `.venv\Scripts\activate`.

Installez [Ollama](https://ollama.com/download), démarrez-le puis téléchargez le modèle local :

```bash
OLLAMA_CONTEXT_LENGTH=8192 ollama serve
# Dans un autre terminal :
ollama pull qwen3-vl:4b-instruct
```

Le modèle multimodal accepte les images et les outils. Il nécessite plusieurs Go de stockage et de mémoire. OpenCLIP télécharge également ses poids au premier lancement. CUDA, MPS ou le CPU sont sélectionnés suivant la disponibilité.

Configuration locale par défaut dans `.env` :

```dotenv
LLM_PROVIDER=ollama
LLM_MODEL=qwen3-vl:4b-instruct
LLM_BASE_URL=http://localhost:11434/v1
```

Pour utiliser OpenAI, choisissez `LLM_PROVIDER=openai`, `LLM_MODEL=gpt-4.1-mini` et renseignez `OPENAI_API_KEY` dans `.env`. Dans ce mode, questions, extraits et images sont envoyés à OpenAI et les appels sont facturés selon votre compte. Ne placez aucune clé dans le notebook ou dans Git.

## Documents d’entrée

Téléchargez le document depuis sa [page officielle arXiv](https://arxiv.org/abs/1706.03762) ou utilisez le PDF fourni par votre enseignant. Placez-le à l’un des emplacements suivants :

```text
attention is all you need.pdf
pdfs/attention is all you need.pdf
```

Puis exécutez :

```bash
python prepare_assets.py
```

Le script copie le PDF de la racine vers `pdfs/` si nécessaire, sans supprimer l’original. Il extrait `architecture_transformer.png` depuis la figure 1 du PDF NeurIPS de 11 pages. Si votre édition a une mise en page différente, fournissez cette image vous-même. Aucun fichier existant n’est écrasé.

Les PDF et images d’entrée sont exclus de Git. Le notebook conserve les illustrations pédagogiques intégrées du document fourni ; elles ne sont pas des résultats calculés. Les extractions, bases vectorielles et poids téléchargés sont également exclus.

## Exécuter le notebook

```bash
python -m jupyter lab
```

Ouvrez `rag_multimodal_agentique.ipynb`, sélectionnez le noyau de `.venv` puis redémarrez le noyau et exécutez toutes les cellules dans l’ordre. Lancez Jupyter depuis la racine du TP. Les questions de démonstration utilisent le LLM configuré : une exécution complète effectue plusieurs générations et appels d’outils.

Pour une exécution automatisée :

```bash
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 rag_multimodal_agentique.ipynb
```

## Interface Streamlit

```bash
python -m streamlit run app.py
```

Choisissez RAG direct pour une requête texte et/ou image. Les deux modes agent acceptent ici une question textuelle ; le notebook contient aussi des exemples multimodaux pour l’agent direct. Le serveur charge automatiquement l’index du PDF configuré, ou le construit à la première utilisation.

## Agent et serveur MCP séparés

```bash
# Vérifier la découverte de l’outil, sans charger de modèle :
python agent_mcp.py --check

# Exécuter une question via MCP :
python agent_mcp.py "Pourquoi utiliser plusieurs têtes d’attention ?"
```

`agent_mcp.py` démarre `mcp_server.py` dans un processus distinct et découvre ses outils via le protocole MCP. Le client agent ne charge pas CLIP et ne lit pas ChromaDB. Le serveur renvoie des réponses et métadonnées publiques ; les chemins locaux ne sont pas transmis à l’agent. Le transport stdio est destiné à une utilisation locale.

## Fichiers principaux

| Fichier | Rôle |
| --- | --- |
| `rag_multimodal_agentique.ipynb` | Progression pédagogique et démonstrations |
| `rag_core.py` | Extraction, indexation, recherche et génération partagées |
| `settings.py` | Configuration du fournisseur LLM |
| `app.py` | Interface Streamlit |
| `agent_react.py` | Graphe explicite MIND/BODY, requêtes multimodales et trace ReAct |
| `retriever_tool.py` | Outil de recherche brute texte ou image, sans génération |
| `agent_local.py` | Agent avec accès direct au RAG |
| `mcp_server.py` | Serveur documentaire MCP |
| `agent_mcp.py` | Agent client MCP |
| `prepare_assets.py` | Préparation des documents locaux |
| `tests/test_rag.py` | Tests de recherche et de génération sans fournisseur externe |

## Vérifications et limites

```bash
python -m unittest discover -s tests -v
python agent_mcp.py --check
```

CLIP dispose d’un contexte textuel court et fonctionne mieux avec des requêtes anglaises sur ce document anglais. Sa similarité cosinus n’est pas une probabilité de pertinence. Une recherche par page peut retrouver plusieurs images de la même page. La présence de citations ne garantit pas automatiquement leur exactitude : relisez les sources pour les conclusions importantes. Le test sur GPT-4/MMLU porte volontairement sur une information absente du document. L’édition NeurIPS fournie pour cette validation contient 11 pages et rapporte 41,0 BLEU en anglais-français ; ce nombre diffère des éditions arXiv ultérieures. Les résultats doivent suivre le PDF réellement indexé.

Les PDF scannés demandent un OCR supplémentaire. Changer de modèle CLIP ou de poids exige de reconstruire l’index avec une nouvelle collection. Les questions et les prompts ne remplacent pas une évaluation indépendante de la fidélité des réponses.

## Références techniques

- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [Agents LangChain](https://reference.langchain.com/python/langchain/agents/factory/create_agent)
- [Limite d’appels d’outils](https://reference.langchain.com/python/langchain/agents/middleware/tool_call_limit/ToolCallLimitMiddleware)
- [Qwen3-VL dans Ollama](https://ollama.com/library/qwen3-vl)
- [SDK Python MCP](https://github.com/modelcontextprotocol/python-sdk)

## Résultats de la validation initiale

- 50 cellules de code exécutées dans l’ordre, sans erreur, avec Qwen3-VL 4B Instruct local et un contexte de 8 192 tokens.
- 216 fragments de texte et 14 images/rendus indexés ; recherche de la figure 2 restreinte à sa page de légende, page 4.
- 11 tests automatisés réussis, incluant la conservation des nombres, la recherche par figure et le rejet des réponses vides ou tronquées.
- Agent direct et agent MCP vérifiés avec un appel effectif à l’outil documentaire ; limite de trois appels contrôlée.
- Interface Streamlit testée avec une vraie requête, réponse et sources affichées.

**Limites observées :** certaines générations du modèle 4B se contredisent sur les paramètres du modèle, confondent le rôle de Q/K/V ou inventent une figure 3. Le notebook contient un corrigé documentaire vérifié et une analyse explicite de ces erreurs. Les sorties originales du LLM sont conservées comme résultats expérimentaux ; elles ne doivent pas être prises pour le corrigé. Les illustrations et citations ne valident pas automatiquement une affirmation.

Les 121 cellules du document initial sont conservées dans leur ordre. Les compléments et le corrigé sont ajoutés à la fin. Les chemins locaux des avertissements ont été anonymisés avant publication.

## Correspondance avec les sept étapes de l’énoncé

| Étape | Code et résultat |
| --- | --- |
| 1 — Extraction multimodale | `rag_core.py` : texte intégral et par page, chunks, extraction des images et rendu des pages vectorielles |
| 2 — Espace latent commun | CLIP ViT-B-32, embeddings texte/image de 512 dimensions normalisés |
| 3 — Stockage | ChromaDB persistant, distance cosinus, empreinte SHA-256, métadonnées type/PDF/page/source et réindexation |
| 4 — Retrieval | `retriever_tool.retrouver` accepte un texte **ou** une image connue, renvoie k sources brutes et leurs scores |
| 5 — Outil agentique | `retriever_tool` : schéma typé, description et paramètres de modalité ; aucune génération LLM imbriquée |
| 6 — ReAct / LangGraph | `agent_react.py` : `StateGraph(MessagesState)` composé puis compilé ; MIND = LLM, BODY = ToolNode |
| 7 — Validation | Deux cas réels dans le complément final du notebook ; trace Action → Observation → Réponse finale et tests de contrôle du graphe |

### Exécuter le graphe ReAct explicite

```bash
python agent_react.py "Pourquoi utiliser plusieurs têtes d’attention ?"
python agent_react.py "Compare cette architecture à la Figure 1 du PDF et explique le décodeur avec les textes et les images retrouvés." --image architecture_transformer.png
```

Le modèle choisit les arguments de `retriever_tool`. Pour une demande texte + image, il doit effectuer deux recherches distinctes et confronter les observations. Les images d’entrée sont enregistrées sous identifiant SHA-256 dans `uploads/`, ignoré par Git ; l’outil ne reçoit aucun chemin arbitraire. Les images issues des observations sont transmises au modèle (trois au maximum, redimensionnées pour limiter le contexte), en plus des textes et des légendes.

Le graphe suit `START → MIND → BODY → MIND → … → END`. Le nœud MIND choisit une action ou une conclusion. BODY exécute les appels et ajoute les observations. Une conclusion sans appel documentaire et un quatrième appel sont refusés. La trace associe les actions et observations par identifiant d’appel ; elle n’expose pas le raisonnement interne du modèle.

Dans Streamlit, choisissez **Agent ReAct LangGraph** pour utiliser ce graphe, une image éventuelle et les aperçus des figures récupérées. Les modes précédents restent disponibles. Le serveur MCP expose maintenant les deux outils : `consulter_pdf` (RAG complet) et `retriever_tool` (récupération brute).

Les 16 tests automatisés couvrent notamment le cycle complet du graphe, la nécessité d’une recherche avant conclusion, la borne de trois appels et la validation des identifiants d’images. Les deux cellules de code du nouveau complément sont autonomes et vérifiées séparément des 50 cellules initialement exécutées. Le corrigé de référence et les limites factuelles du LLM restent applicables.

Référence : [construction explicite d’un agent avec LangGraph](https://docs.langchain.com/oss/python/langgraph/quickstart).

Validation du complément : deux appels documentaires dans chacun des deux cas exécutés ; le cas texte + image récupère des textes et des images. Les vecteurs texte et image contrôlés ont 512 composantes et une norme de 1. Le schéma MCP expose bien les deux outils.
