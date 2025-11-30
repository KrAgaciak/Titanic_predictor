# Projekt przewidywania prawdopodobieństwa przeżycia katastrofy Titanica

## Źródło danych

Dane zostały pobrane z publicznego zbioru **Titanic Dataset** dostępnego na platformie Kaggle:

Źródło: https://www.kaggle.com/datasets/yasserh/titanic-dataset
Data pobrania: 14.10.2025
Licencja: Open Data / Public Domain

## Opis zbioru danych

Zbiór danych zawiera informacje o pasażerach statku RMS Titanic, m.in.:

| Kolumna       | Opis                                |
|---------------|-------------------------------------|
| `Survived`    | Czy pasażer przeżył (1) lub nie (0) |
| `Pclass`      | Klasa biletu (1, 2, 3)              |
| `Sex`         | Płeć pasażera                       |
| `Age`         | Wiek pasażera                       |
| `Fare`        | Cena biletu                         |
| `Embarked`    | Port wejścia na pokład              |

Dane będą użyte w zadaniu klasyfikacji – **predykcja przeżycia pasażera (Survived)**.

## Metryka ewaluacji

Wybrana metryka: **F1-score**

**Uzasadnienie:**
Dane są niezbalansowane (różna liczba osób, które przeżyły / nie przeżyły).
F1-score lepiej ocenia model niż dokładność (accuracy),
ponieważ uwzględnia zarówno precyzję (precision), jak i czułość (recall).

## KEDRO QUICKSTART

cd \Titanic_ASI\titanic_predictor

conda activate asi-ml

kedro run
# lub

kedro run --pipeline data_science

wandb login KEY

## W&B

Tabela runów
<img width="1678" height="448" alt="obraz" src="https://github.com/user-attachments/assets/c53f0288-d8ae-4aa9-bd6a-d07fb12f7457" />

Porównanie modeli
<img width="884" height="296" alt="obraz (1)" src="https://github.com/user-attachments/assets/cc3b8857-fabe-46c4-aab8-7ad15bc5fa93" />


