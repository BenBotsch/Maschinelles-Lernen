# Maschinelles Lernen: Code zum Fachbuch

![Python](https://img.shields.io/badge/python-3-3776ab)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-ff6f00)
![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c)

Dieses Repository enthält die Python-Beispiele zum Buch *Maschinelles Lernen. Grundlagen und Anwendungen. Mit Beispielen in Python* (Springer Spektrum, 2023). Jedes Verzeichnis gehört zu einem Kapitel, und jedes Skript lässt sich einzeln ausführen.

<img src="https://user-images.githubusercontent.com/39590031/229341924-7b64acbf-7909-4474-9e45-0bc3845182ba.png" alt="Cover des Buchs Maschinelles Lernen" width="220">

| Angabe | Wert |
|---|---|
| ISBN Druck | 978-3-662-67276-1 |
| ISBN E-Book | 978-3-662-67277-8 |
| DOI | [10.1007/978-3-662-67277-8](https://doi.org/10.1007/978-3-662-67277-8) |

## Inhalt

| Kapitel | Beispiele |
|---|---|
| [2 Lineare Algebra](2_Lineare_Algebra/) | Numerische Lösungsverfahren |
| [3 Wahrscheinlichkeit und Statistik](3_Wahrscheinlichkeit_und_Statistik/) | Schiefe von Verteilungen |
| [4 Optimierung](4_Optimierung/) | Gradientenabstieg |
| [5 Parametrische Methoden](5_Parametrische_Methoden/) | Regression, Bayes, LDA, SVM, mehrschichtiges Perzeptron |
| [6 Nichtparametrische Methoden](6_Nichtparametrische_Methoden/) | k-nächste Nachbarn, Entscheidungsbaum |
| [7 Bestärkendes Lernen](7_Bestärkendes_Lernen/) | Reinforcement Learning |
| [8 Clusteranalyse](8_Custeranalyse/) | k-Means |
| [9 Anwendungen](9_Anwendungen/) | Bildklassifikation, Objekterkennung, Grad-CAM, GAN, RNN, Übersetzung mit Transformer, Mini-GPT, Spracherkennung mit CTC, Molekülvorhersage, Regelung und weitere |

Gemeinsame Hilfsfunktionen für Datensätze, Metriken und Abbildungen liegen in [Utils/](Utils/), ein Beispieldatensatz in [Datasets/](Datasets/).

## Ausführen

Die Skripte nutzen vor allem NumPy, pandas, scikit-learn, Matplotlib und TensorFlow mit Keras. Einzelne Anwendungen brauchen zusätzlich PyTorch, Gym, RDKit, KerasNLP oder jiwer.

```sh
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install numpy pandas scikit-learn matplotlib seaborn scipy tensorflow
python 5_Parametrische_Methoden/regression.py
```

Die Skripte entstanden 2023 mit den damals aktuellen Bibliotheken. Bei neueren Versionen kann eine Anpassung nötig sein.

## Zitieren

```bibtex
@book{botsch2023ml,
  author    = {Botsch, Benny},
  title     = {Maschinelles Lernen. Grundlagen und Anwendungen},
  subtitle  = {Mit Beispielen in Python},
  publisher = {Springer Spektrum},
  year      = {2023},
  doi       = {10.1007/978-3-662-67277-8}
}
```
