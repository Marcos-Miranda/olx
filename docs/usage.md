# Guia de uso para o cientista de dados

## O que você fornece

Um diretório de projeto com:

- `pipeline.py`: os steps e o `Pipeline`. É o único código que você escreve.
- `requirements.txt` (opcional): as dependências do pipeline.
- Os dados. Eles podem ser lidos de qualquer lugar no primeiro step, ou ficar dentro do diretório do projeto, como o `iris.csv` do exemplo.

Não precisa escrever Dockerfile, servidor ou rotas. O exemplo completo está em [examples/iris/](../examples/iris/).

## Escrevendo o pipeline

```python
from pathlib import Path

from pydantic import BaseModel
from mlbox import Pipeline, get_context, step


class InputParams(BaseModel):
    dataset_path: str = "iris.csv"
    n_estimators: int = 100


@step()
def preprocess() -> dict[str, Path]:
    import pandas as pd

    ctx = get_context(InputParams)
    df = pd.read_csv(ctx.input_params.dataset_path)
    ...
    return {"train": train_path, "test": test_path}


@step(cpus=2, memory="2g")
def train(train_path: str) -> Path: ...


@step()
def evaluate(test_path: str, model_path: str) -> dict[str, float]:
    ctx = get_context(InputParams)
    ...
    ctx.mlflow.log_metric("accuracy", accuracy)
    ctx.register_model(model, input_data=features, predict_function="predict_proba")
    return {"accuracy": accuracy}


data = preprocess()
model = train(data["train"])
report = evaluate(data["test"], model)

pipeline = Pipeline("iris", InputParams, steps=[evaluate], requirements="requirements.txt")
```

### Steps

`@step(cpus=1, memory="1g")` transforma uma função em um step, que roda no seu próprio container. `cpus` e `memory` são os limites do container (os mesmos do `docker run --cpus` e `--memory`).

Chamar um step no nível do módulo não executa nada. A chamada retorna uma referência para o valor que o step vai retornar, e essa referência pode ser indexada (`data["train"]`). Passar uma referência para outro step cria uma dependência: o segundo só roda depois que o primeiro terminar com sucesso, e recebe o valor. Steps independentes rodam em paralelo.

Regras:

- Argumentos precisam ser literais JSON (`dict`, `list`, `int`, `float`, `str`, `bool`, `None`) ou referências.
- O retorno precisa ser serializável em JSON. `Path` é aceito e vira string.
- Cada step pode ser chamado uma vez só, e o nome do step é o nome da função.
- Arquivos (datasets, modelos) passam de um step para outro pelo `ctx.data_path`: um step grava o arquivo e retorna o caminho.
- O arquivo do pipeline também é importado na sua máquina pela CLI. Por isso, bibliotecas que só estão no `requirements.txt` devem ser importadas dentro dos steps.

### Pipeline

`Pipeline(name, parameters, steps, requirements=None)`:

- `name`: nome do experimento no MLflow e do modelo registrado.
- `parameters`: um `BaseModel` do pydantic com os parâmetros do pipeline. Eles são validados antes de qualquer coisa rodar.
- `steps`: basta passar os steps finais, os anteriores entram automaticamente.
- `requirements`: caminho do requirements, relativo ao arquivo do pipeline. Fixe as versões para o ambiente ser reprodutível.

### Context

Dentro de um step, `get_context(InputParams)` retorna:

| Atributo | Descrição |
| --- | --- |
| `input_params` | Os parâmetros da execução, validados pelo `InputParams` |
| `data_path` | Diretório compartilhado por todos os steps da execução |
| `random_seed` | A seed da execução. `random` e NumPy já são inicializados com ela, mas passe para os estimadores também (`random_state=ctx.random_seed`) |
| `mlflow` | O módulo `mlflow`, com a run da execução já ativa. `mlflow.log_metric(...)` direto também funciona |
| `register_model(...)` | Registra o modelo para ser servido (abaixo) |

O diretório do projeto é o working directory do step, então caminhos relativos (`iris.csv`, `data/train.csv`) funcionam.

### Registrando o modelo

Esse é o contrato para servir o modelo. Em algum step, chame:

```python
ctx.register_model(model, input_data=features, predict_function="predict_proba")
```

- `model`: qualquer objeto treinado, de qualquer framework.
- `predict_function`: o nome do método que faz a predição (`"predict"`, `"predict_proba"`, ...). Ele recebe um DataFrame e deve retornar números: um valor por instância (regressão) ou um vetor por instância (probabilidades).
- `input_data`: um DataFrame com as features. O schema da API é inferido dele: nomes, tipos e quais colunas são obrigatórias (uma coluna com nulos vira opcional). As primeiras linhas viram o exemplo da documentação da API.

O modelo é registrado no MLflow com o nome do pipeline, e cada chamada cria uma nova versão.

## Rodando o pipeline

Setup, uma vez (requer Docker, [uv](https://docs.astral.sh/uv/) e Python 3.13):

```bash
uv sync                       # instala a CLI
make build                    # imagem base mlbox:latest
docker compose up -d mlflow   # MLflow em http://localhost:5001
```

Execução:

```bash
uv run mlbox run examples/iris/pipeline.py -p n_estimators=50 --seed 7
```

- `-p key=value` define um parâmetro. O valor é lido como JSON quando possível (`50`, `0.3`) e como string caso contrário.
- Sem `--seed`, uma seed é gerada. Ela é registrada na run de qualquer forma.
- Os logs dos steps aparecem com o nome do step como prefixo. A CLI mostra o run ID no começo e o status no final.
- Se um step falha, o erro aparece no terminal, os steps que não começaram são pulados (`skipped`) e a run termina como `FAILED`.

Na primeira execução a imagem do pipeline é construída com o `requirements.txt`. Ela é reaproveitada enquanto o requirements e a imagem base não mudam.

## Vendo os resultados

- **MLflow UI** (<http://localhost:5001>): experimento com o nome do pipeline, com parâmetros, seed, métricas, o modelo e o status de cada step (tags `mlbox.step.<nome>.status`).
- **CLI**: `uv run mlbox list` lista as runs, e `uv run mlbox show <run_id>` mostra parâmetros, seed, imagem e o status de cada step.
- **Arquivos dos steps**: o que foi gravado em `ctx.data_path` fica em `.mlbox/runs/<run_id>/data`, no diretório onde a CLI rodou.

## Servindo o modelo

```bash
uv run mlbox serve --model iris                   # última versão, em http://localhost:8080
uv run mlbox serve --model iris --version 2 --port 9000 --workers 2
```

- Sem `--version`, serve a última versão (a de maior número).
- Argumentos extras vão para o uvicorn (`--workers 2`).
- A API roda em um container da mesma imagem que treinou o modelo, então tem as mesmas dependências.
- O comando fica em primeiro plano até o Ctrl+C, que para e remove o container.
- Se o `make up` estiver rodando, a porta 8080 já está em uso. Use `--port`.

### Endpoints

| Endpoint | Descrição |
| --- | --- |
| `GET /health` | `{"status": "ok"}` |
| `GET /info` | Nome, versão e schema de input do modelo |
| `POST /predict` | Predições |
| `GET /docs` | Swagger, com o exemplo de input do modelo já preenchido |

O `/predict` recebe um array JSON com uma ou mais instâncias e retorna uma lista de números por instância:

```bash
curl -X POST localhost:8080/predict -H 'Content-Type: application/json' \
  -d '[{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
       {"sepal_length": 6.2, "sepal_width": 3.4, "petal_length": 5.4, "petal_width": 2.3}]'
```

```json
[[0.9604, 0.0395, 0.0000039], [0.00094, 0.1006, 0.8984]]
```

O input é validado contra o schema do modelo e erros retornam HTTP 422:

- Instância sem uma feature obrigatória, valor de tipo errado (string em uma feature numérica) ou array vazio são rejeitados.
- `null` é aceito, e o modelo decide o que fazer com ele.
- Um inteiro é aceito em uma feature `double`.

## Referência da CLI

| Comando | Descrição |
| --- | --- |
| `mlbox run <arquivo> [-p key=value ...] [--seed N]` | Valida os parâmetros, constrói a imagem e roda o pipeline |
| `mlbox list [--pipeline NOME]` | Lista as runs, das mais novas para as mais antigas |
| `mlbox show <run_id>` | Mostra os detalhes de uma run |
| `mlbox serve --model NOME [--version N] [--port PORTA] [ARGS_UVICORN...]` | Serve um modelo registrado |

A CLI acessa o MLflow em `MLFLOW_TRACKING_URI` (padrão `http://localhost:5001`).
