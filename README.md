# mlbox

Ferramenta para gerenciar o ciclo de vida de modelos de ML para cientistas de dados que escrevem apenas Python. O cientista escreve um pipeline com funções Python (steps), e o mlbox:

- roda cada step em um container isolado, com seus próprios limites de CPU e memória;
- registra parâmetros, seed, métricas, artefatos e modelo de cada execução no MLflow;
- serve o modelo treinado em uma API REST, sem o cientista escrever servidor, rotas ou Dockerfile.

## Rodando o exemplo

### Pré-requisitos

- Docker com Docker Compose.
- `make` (no macOS vem com o Xcode Command Line Tools).
- Portas 5001 (MLflow) e 8080 (API) livres.

### Treinar e servir

```bash
make up
```

O comando constrói as imagens, sobe o MLflow, treina o pipeline de exemplo ([examples/iris/](examples/iris/)) e serve a última versão do modelo registrado. Ele retorna quando o treino termina. Para acompanhar os logs:

```bash
docker compose logs -f
```

O pipeline de exemplo usa o dataset iris e tem 4 steps: `preprocess` divide os dados, `train_random_forest` e `train_logistic_regression` rodam em paralelo, e `evaluate` loga a acurácia dos dois e registra o melhor:

```python
data = preprocess()
random_forest = train_random_forest(data["train"])
logistic_regression = train_logistic_regression(data["train"])
report = evaluate(data["test"], random_forest, logistic_regression)

pipeline = Pipeline("iris", InputParams, steps=[evaluate], requirements="requirements.txt")
```

A run aparece no experimento `iris` do [MLflow](http://localhost:5001), com parâmetros, seed, métricas, modelo e o status de cada step. Os arquivos gravados pelos steps ficam em `.mlbox/runs/<run_id>/data`.

### Chamar a API

A documentação interativa fica em <http://localhost:8080/docs>, com o exemplo de input do modelo já preenchido no `/predict`. Pelo terminal:

```bash
curl -X POST localhost:8080/predict -H 'Content-Type: application/json' \
  -d '[{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
       {"sepal_length": 6.2, "sepal_width": 3.4, "petal_length": 5.4, "petal_width": 2.3}]'
```

A resposta tem uma lista de probabilidades (uma por classe) para cada instância:

```json
[[0.9604, 0.0395, 0.0000039], [0.00094, 0.1006, 0.8984]]
```

`GET /info` mostra o nome, a versão e o schema de input do modelo, e `GET /health` retorna `{"status": "ok"}`.

### Parar

```bash
make down    # para os serviços
make clean   # remove tudo o que o make up criou: containers, rede, dados do MLflow, imagens do mlbox e .mlbox
```

## Testes

Requer o [uv](https://docs.astral.sh/uv/) e Python 3.13 ou mais novo.

```bash
uv run pytest
```

Os testes não precisam de Docker nem do MLflow rodando.

## Documentação

- [usage.md](docs/usage.md): guia do cientista de dados. O que ele precisa fornecer e como usa a CLI para treinar, ver resultados e servir.
- [library.md](docs/library.md): arquitetura da library e como treino e serving funcionam por dentro.
- [DECISIONS.md](docs/DECISIONS.md): decisões técnicas, alternativas descartadas, o que ficou fora do escopo e o tempo gasto.
- [AI_USAGE.md](docs/AI_USAGE.md): como usei IA no desenvolvimento.
- [architecture.md](docs/architecture.md): como a solução funcionaria em produção na AWS.
