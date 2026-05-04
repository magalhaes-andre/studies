# Classificacao de Cancer de Pulmao

Projeto simples em `TensorFlow/Keras` para classificar imagens de CT do dataset **IQ-OTH/NCCD**.

- entrada: imagem JPG
- saida: probabilidade de `maligno`
- classes usadas no dataset:
  - `Normal cases`
  - `Bengin cases`
  - `Malignant cases`

O arquivo principal: `lung_cancer_detection_workflow.ipynb`.

## Dataset

Estrutura esperada:

```
The IQ-OTHNCCD lung cancer dataset/
  Bengin cases/
  Malignant cases/
  Normal cases/
```

Caminho padrao atual:
```
C:\Users\Usuario\Documents\Lung Cancer\The IQ-OTHNCCD lung cancer dataset\The IQ-OTHNCCD lung cancer dataset\The IQ-OTHNCCD lung cancer dataset\The IQ-OTHNCCD lung cancer dataset
```

## Instalacao

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Como Executar

1. Abra `lung_cancer_detection_workflow.ipynb`
2. Ajuste `DATASET_DIR` e `OUTPUT_DIR` se necessario
3. Rode as celulas de cima para baixo
4. O treino usa o dataset completo definido em `DATASET_DIR`

## Arquivos Gerados

Depois do treino, o notebook salva:

- `best.keras`
- `best_info.json`
- `metrics.json`
- `train_records.csv`
- `val_records.csv`
- `test_records.csv`

## Inferencia

Depois do treino, use a funcao `prever_imagem(...)` no notebook para obter a probabilidade de malignidade de uma imagem.

## Observacoes

- Este projeto e academico.
- O split e por imagem, nao por paciente.
- A saida do modelo e uma probabilidade estimada pelo modelo, nao um diagnostico medico definitivo.
