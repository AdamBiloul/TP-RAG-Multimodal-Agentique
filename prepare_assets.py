"""Préparer les entrées publiques du TP sans écraser les fichiers existants."""
from pathlib import Path
import shutil
import pymupdf
from settings import ROOT


def preparer():
    destination = ROOT / 'pdfs' / 'attention is all you need.pdf'
    original = ROOT / 'attention is all you need.pdf'
    if not destination.is_file():
        if not original.is_file():
            raise FileNotFoundError('Ajoutez attention is all you need.pdf à la racine du TP ou dans pdfs/.')
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)
    image = ROOT / 'architecture_transformer.png'
    if not image.is_file():
        with pymupdf.open(destination) as document:
            # Figure 1 du PDF NeurIPS (11 pages) fourni pour le TP.
            page = document[2]
            if 'Figure 1:' not in page.get_text():
                raise ValueError('Mise en page différente : fournissez architecture_transformer.png manuellement.')
            page.get_pixmap(matrix=pymupdf.Matrix(2, 2),
                           clip=pymupdf.Rect(180, 65, 435, 420)).save(image)
    print('Entrées disponibles :', destination.name, 'et', image.name)


if __name__ == '__main__':
    preparer()
