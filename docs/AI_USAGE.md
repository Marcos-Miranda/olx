# Uso de IA

## Ferramenta

Usei o Claude Code, que é a ferramenta que uso no dia a dia.

## Entendendo o SageMaker

Como não tenho experiência prática com o SageMaker, usei o Claude para entender melhor como ele funciona antes de desenhar a ferramenta, já que ela é baseada nos conceitos dele:

```text
Mas tenho dúvidas do funcionamento do sagemaker, pra me basear melhor:
- Como lida com requirements de cada step?
- Como funciona hyperparameter tuning?
- Como parâmetros/hiperparâmetros são passados pra uma run?
- poderia criar um pipeline completo do sagemaker? com um exemplo bem minimalista
```

Isso ajudou a decidir quais conceitos trazer para a ferramenta e quais deixar de fora (ver [DECISIONS.md](DECISIONS.md#baseado-no-sagemaker-sem-copiar)).

## Spec driven development

Antes de começar o desenvolvimento de fato, criei um framework de spec driven development para trabalhar com a ferramenta de forma organizada e estruturada, sempre garantindo que os requisitos estão bem alinhados antes do código. É como prefiro interagir com essas ferramentas. É uma versão bem minimalista, bem menos completa do que a que uso no dia a dia, mas serviu para demonstrar a metodologia e funcionou muito bem.

Cada tarefa vira uma *track* em [tracks/](../tracks/), com duas skills:

- [`/new-track <descrição>`](../.claude/skills/new-track/SKILL.md): tira dúvidas sobre a tarefa, escreve o `spec.md` (requisitos funcionais e critérios de aceite) e espera minha revisão. Depois escreve o `plan.md` (tarefas em blocos, com o teste antes da implementação e uma verificação final), espera minha revisão de novo e registra a track em [registry.md](../tracks/registry.md).
- [`/implement-track [track_id]`](../.claude/skills/implement-track/SKILL.md): implementa o plano tarefa por tarefa seguindo TDD, pede minha revisão e faz um commit a cada bloco, e no final roda a verificação final e pede a revisão final.

O [AGENTS.md](../AGENTS.md) completa com as regras do projeto: uso do uv, ruff e mypy depois de cada implementação, docstrings e como escrever testes.

## Exemplos de interações

### Criação do framework

<details>
<summary>Prompt</summary>

````text
now I want to do the following: implement a MINIMAL version of spec driven development workflow, which is how I would work in a real scenario. What I want for that:

- The skill new-track: Given a description of a taks it should:
    - ask for more information if there is something not clear
    - after understading the task, it should:
        - create an ID for the track: <name1>_<name2>_<today-date-only-numbers>
        - create a folder in /tracks with the ID as the name
        - create a spec.md file in that folder contatining the functional requirements and acceptance criteria for the task
        - ask for the user to review the spec: accept or suggest changes
        - after the spec is accepted, create plan.md, which should contain:
            - a plan of exactly what the agent should implement to met the spec
            - the plan should be readable by a human, so it should contain a lot of code and cluttered information
            - the tasks can be broken into coherent blocks
            - each task contain a checkbox
            - tasks that need tests describe the test before the implementation (follows TDD)
            - the is a final section with final verification: linters, tests, acceptance criteria, with checkboxes
        - ask for the user to review the plan : accept or suggest changes
        - after accepted: registry the track on /tracks/registry.md and commit: "feat: started track <track_id>"

- The skill implement-task: can receive a track_id, if none is passed read /tracks/registry.md to continue the track that is in_progress or not_started:
- It should:
    - mark the track as in_progress in the registry, if not already
    - read spec.md and plan.md related to the track 
    - start implement the tasks that are not completed ([~] or [ ])
    - before implementing the task, mark it as in progress [~]
    - after implementing the task mark as completed [X]
    - after each block is completed, ask for user review, after accepted commit: feat: finished block x of track <track_id>
    - after all tasks are completed run the final verification and check everything that is met.
    - ask for final human revier and after approval: mark the track completed in the registry and commit: feat: finished track <track_id>

be as minimal as possible in the skills
````

</details>

**Resultado**: ficou muito bom, só tive que ajustar uma ou outra coisa no template de spec e plan das skills.

### Track 1: pipeline de treinamento

<details>
<summary>Prompt</summary>

````text
/new-track A tarefa é construir a ferramenta "mlbox", que seria uma library composta por um SDK e CLI e baseadas em alguns conceitos do Amazon SageMaker:

## SDK

Seria centrada nos seguintes componentes:

### Step

Equivale a um step em um pipeline de Machine Learning, como preprocess, train, evaluate, etc. Cada step é executado isoladamente em um container.

- Classe `Step`
- decorator `@step(cpu: int, memory: str)`:  Onde cpu e memory são os limites de recursos do container. Exemplo: (cpu=1, memory="4g")
- funções decoradas deveriam receber e retornar apenas tipo compatíveis com JSON (dict, list, int, float, str, bool)

### Context

Uma classe que gerencia informações e estado durante a execução do pipeline. A minha ideia inicial era que usuário pudesse chamar `ctx = get_context()` dentro de um step, mas fico aberto à sugestões.
Métodos e atributos que deveria ter:
- `.input_params`: BaseModel. Um objeto do pydantic contendo os parâmetros passado na execução do pipeline. O BaseModel seria definido na construção do pipeline. O ideal seria que Context fosse genérico no BaseModel, mas se for muito complexo deixar como Any é ok.
- `.data_path`: pathlib.Path. Caminho para um diretório para o qual todos os steps terão acesso, podem gravar e ler dados nele. Cada execução recebe um diretório único.
- `.mlflow`: uma instância do MLFlow (podendo ser usada para .log_param ou .log_metric por exemplo) dentro de uma run já iniciada no começo do pipeline. O nome do experimento seria o nome do Pipeline.
- `.random_seed`: a ser usado em todos os steps para garantir reproducibilidade.

### Pipeline

Define a configuração e steps do pipeline

Pipeline(name: str, parameters: type[BaseModel], steps: list[Step], requirements: str)


### Runner

Executa os steps em containers, orquestrando sequência e paralelismo. O grafo de execução é definido por dependência entre outputs e inputs de steps.

## Exemplo    

```python
# my-mode/pipeline.py

class InputParams(BaseModel):
    dataset_path: Path
    n_estimators: int = Field(default=100, gt=0)

@step(cpus=1, memory="2g")
def preprocess() -> dict[str, str]:
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    ctx = get_context()
    df = pd.read_csv(ctx.input_params.dataset_path)
    train, test = ...
    train_path = ctx.data_path / "train.csv"
    test_path = ctx.data_path / "test.csv"
    train.to_csv(train_path)
    test.to_csv(test_path)
    return {"train": train_path, "test": test_path}

@step(cpus=2, memory="4g")
def train(train_path: str) -> str:
    import pandas as pd
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    
    ctx = get_context()
    train = pd.read_csv(train_path)
    model = RandomForestClassifier(
        n_estimators=ctx.input_params.n_estimators,
        random_state=ctx.random_seed
    ).fit(train.drop(columns="target"), train["target"])
    model_path = ctx.data_path / "model.joblib"
    joblib.dump(model, model_path)
    mlflow.log_param("n_estimators", ctx.input_params.n_estimators)
    return model_path

data = preprocess() # do not run yet
model = train(data["train"]) # identify order dependency

Pipeline(name="my-model", parameters=InputParams, steps=[train], requirements="./requirements.txt")
```

## CLI

Usado para executar e listar runs de pipelines (incluindo o status). Parâmetros do pipeline podem ser passados na execução.

Deve ser criada um Dockerfile contendo a library e ela deve ser usada para executar os steps. Os requirements do pipeline devem ser instalados antes da execução.

Não se esqueça que isso é projeto para um case, não algo que será colocado em produção ou usado na prática.
````

</details>

**Resultado**: muito bom no geral também ([spec](../tracks/mlbox_pipelines_20261021/spec.md) e [plan](../tracks/mlbox_pipelines_20261021/plan.md)). Só tive que mudar duas coisas no plano:

```text
2 mudanças:
- mude a porta exposta do mlflow para 5001. 5000 é problemática no macos
- não coloque a doc no README, vou usá-lo com outro propósito depois. Coloque em docs/project.md
```

E tive que corrigir um bug no commit seguinte ([43db8bd](https://github.com/Marcos-Miranda/olx/commit/43db8bd0838ab9abd2f8059d402aebbf93d30c2d)): o `build_pipeline_image` não reconstruía a imagem do pipeline quando a imagem base mudava, porque a tag era só o hash do `requirements.txt`. Agora o ID da imagem base também entra no hash.

### Track 2: serving API

<details>
<summary>Prompt</summary>

````text
/new-track quero adicionar uma nova funcionalidade no projeto: servir um modelo registrado no MLFlow atráves de uma REST API com FastAPI

O que imagino:
- Um novo comando na CLI: `mlbox serve --model <registered-model-name> --version <registered-model-version>` , em que --version seria opcional, se não passado usamos a latest. Parâmetros extras são passado diretamente pro comando uvicorn.
- O comando deveria usar a mesma imagem usada no treinamento pra garantir consistência de dependências. O que implica que teremos que logar a tag da imagem na run de treinamento

Sobre a API:
- carrega o modelo do MLFlow no lifespan
- armazena informações do modelo, como input schema
- seta os exemplos de input do modelo como exemplos no json schema do endpoint predict
- 3 endpoints:
    - /health: apenas responde ok
    - /info: informações do modelo: nome, versão, e input schema
    - /predict:  roda a inferência do modelo. O input deve aceitar um ou mais inputs e ser genérico em relação às features aceitas: list[dict[str, float]]. A própria validação do MLFlow vai ser responsável pela validação do input (transformar MlflowException em HTTP_422). O output pode ser list[list[float]] para possiblitar tanto valor único (como em regressão) como probabilidades.
- injetar modelo/infos nos endpoints como dependencies pra facilitar testes
````

</details>

**Resultado**: sem problemas também ([spec](../tracks/model_serving_20261022/spec.md) e [plan](../tracks/model_serving_20261022/plan.md)). Só algumas interações normais durante a revisão da spec e do plano:

```text
# revisão da spec
há algo a ser fixado: deveríamos aceitar outros tipos para feature values, como int, bool and str

# revisão do plano
- seria interessante teste como tipo inválido no input também. Exemplo: uma feature float recebeu uma string
```

## Revisão durante a implementação

Durante a implementação das tracks eu revisava o código depois de cada bloco, e sempre concordei com o que foi feito. Fiz só um comentário, na track 2:

```text
health and info should be async, only predict should not so FastAPI runs it in its worker threadpool instead of blocking the event loop on CPU-bound work
```

Na revisão final da track 2, testando a API, encontrei dois problemas no código gerado e pedi a correção:

```text
I am testing the server and there is some problems:
- nulls are not accepted by the pydantic schema. They shold because the models can deal with it.
- when I pass more than one instance, if one column is missing it gets the None value because of pd.DataFrame(body.root) and mlflow does not raise an error. for a required coluimn Which is inconsistent with one instance with a missing feature
```
