"""Outil de récupération brute : aucune génération LLM dans le retriever."""
import hashlib
import json
import re
from pathlib import Path
from typing import Literal
from PIL import Image
from langchain.tools import tool
from settings import ROOT


def enregistrer_image(chemin):
    """Copier une image de requête validée dans le stockage local ignoré par Git."""
    from io import BytesIO
    with Image.open(chemin) as image:
        tampon = BytesIO()
        image.convert('RGB').save(tampon, format='PNG')
    donnees = tampon.getvalue()
    identifiant = hashlib.sha256(donnees).hexdigest()
    dossier = ROOT / 'uploads'
    dossier.mkdir(exist_ok=True)
    destination = dossier / f'{identifiant}.png'
    if not destination.exists():
        destination.write_bytes(donnees)
    return 'input:' + identifiant


def retrouver(texte=None, image_id=None, type_resultat='tous', k=6):
    """Récupérer les sources brutes à partir d'un texte OU d'une image connue."""
    if (texte is None) == (image_id is None):
        raise ValueError('Fournissez soit texte, soit image_id.')
    if texte is not None and (not isinstance(texte, str) or not texte.strip()):
        raise ValueError('Le texte doit être non vide.')
    if type(k) is not int or not 1 <= k <= 12:
        raise ValueError('k doit être un entier entre 1 et 12.')
    if type_resultat not in {'tous', 'texte', 'image'}:
        raise ValueError('Type de résultat invalide.')
    if image_id is not None and not isinstance(image_id, str):
        raise ValueError('Identifiant image invalide.')
    from rag_core import initialiser, rechercher, image_source
    chemin = None
    if image_id is not None and image_id.startswith('input:'):
        empreinte = image_id.removeprefix('input:')
        if not re.fullmatch(r'[a-f0-9]{64}', empreinte):
            raise ValueError('Identifiant image invalide.')
        chemin = ROOT / 'uploads' / f'{empreinte}.png'
        if not chemin.is_file():
            raise FileNotFoundError('Image de requête inconnue. Enregistrez-la avant la recherche.')
    initialiser()
    if image_id is not None and chemin is None:
        chemin = image_source(image_id)
        if chemin is None:
            raise ValueError('Cet identifiant ne désigne pas une image indexée.')
    resultats = rechercher(texte=texte, image=chemin,
        type_resultat=None if type_resultat == 'tous' else type_resultat, k=k)
    sources = []
    for identifiant, contenu, metadata, distance in zip(
        resultats['ids'][0], resultats['documents'][0],
        resultats['metadatas'][0], resultats['distances'][0]):
        sources.append({
            'id': identifiant, 'type': metadata['type'], 'pdf': metadata['pdf'],
            'page': metadata['page'], 'similarite': 1 - distance,
            'contenu': contenu, 'legendes_page': metadata.get('legendes_page', ''),
        })
    return {'requete': {'texte': texte, 'image_id': image_id}, 'sources': sources}


@tool
def retriever_tool(texte: str | None = None, image_id: str | None = None,
                   type_resultat: Literal['tous', 'texte', 'image'] = 'tous', k: int = 6) -> str:
    """Rechercher les k sources brutes du PDF, sans génération de réponse.

    Fournir exactement une requête : texte (anglais conseillé) OU image_id.
    image_id est fourni par l'utilisateur ou provient d'un résultat image précédent.
    Ne jamais inventer un identifiant et ne jamais fournir un chemin de fichier.
    Choisir tous pour mélanger textes et images, texte pour les passages, image
    pour les figures. Conserver Figure 1/2 dans les requêtes de figures numérotées.
    Les légendes concernent la page entière ; elles ne prouvent pas les pixels
    d'une image isolée. La similarité cosinus n'est pas une probabilité.
    """
    return json.dumps(retrouver(texte, image_id, type_resultat, k), ensure_ascii=False)
