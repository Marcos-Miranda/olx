# Decisões técnicas

O foco foi ter o fluxo ponta a ponta funcionando (treinar, registrar artefatos e métricas, servir) com algo que se pareça com uma ferramenta de verdade. Preocupações de produção ficaram de lado de propósito e estão listadas em [Fora do escopo](#fora-do-escopo).

## Baseado no SageMaker, sem copiar

A OLX usa SageMaker, então peguei os conceitos dele que mais ajudam a resolver o problema:

- Cada step é um job isolado, em um container próprio e com seus próprios recursos.
- O grafo é inferido pelas dependências de dados entre os steps, como no `@step` do SageMaker Pipelines: um step que usa a saída de outro espera por ele, e steps independentes rodam em paralelo.
- O container segue o layout `/opt/ml` (hiperparâmetros em `/opt/ml/input/config`, falha em `/opt/ml/output/failure`), e a API roda na porta 8080.

Não copiei a interface nem segui os contratos à risca. `/opt/ml/data` e `/opt/ml/output/result.json` não existem no SageMaker, o modelo vai para o MLflow em vez de `/opt/ml/model` (`model.tar.gz`), os recursos são `cpus`/`memory` em vez de `instance_type`, e a API expõe `/health` e `/predict` em vez de `/ping` e `/invocations`. O SageMaker entra no desenho de arquitetura como a versão de produção da ferramenta. O objetivo aqui não é que a mesma imagem rode nos dois lugares.

Descartei o Local Mode do SDK do SageMaker (`LocalPipelineSession`), que roda pipelines em Docker local, por dois motivos: ele exige configuração da AWS (credenciais, região, imagens no ECR), então o avaliador não conseguiria rodar com um comando, e ele faria justamente o que está sendo avaliado.

## Abstração para o cientista

### Treino: steps como funções Python

O cientista escreve funções com `@step` e um `Pipeline` em um arquivo `.py`, e entrega junto o `requirements.txt` e os dados. Dentro do step o código é livre: qualquer biblioteca e qualquer chamada do MLflow. O contrato completo está no [guia de uso](usage.md).

Alternativas consideradas:

- **Script livre + SDK** (`mlbox.log_model(model)`): dá liberdade máxima, mas a ferramenta só descobre no final se o modelo foi registrado e não consegue separar etapas.
- **Uma função de entrada `train(ctx)`**: simples, mas o treino inteiro vira uma etapa só. Com steps, cada etapa pede seus recursos (o pré-processamento pode precisar de mais memória e o treino de mais CPU) e etapas independentes rodam em paralelo. Um pipeline com um step só é equivalente.
- **Classe com `train()` e `predict()`** (estilo BentoML, Truss, Cog): mais cerimônia do que o necessário.

Outras escolhas do contrato:

- **`get_context()` em vez de receber `ctx` como argumento**: evita tratar um argumento que foge do padrão de referências e dependências entre steps. Receber `ctx` como argumento seria uma experiência melhor para o cientista, mas exigiria esse tratamento à parte.
- **Argumentos e retornos JSON**: os valores trocados ficam explícitos e podem ser inspecionados (`step.json` e `result.json`). Dados grandes passam por arquivos no `data_path`.
- **Parâmetros como `BaseModel` do pydantic**: validados pela CLI antes de qualquer coisa rodar, e tipados dentro do step com `get_context(InputParams)`.

### Serving: `ctx.register_model`

O contrato mínimo para servir é chamar, em algum step:

```python
ctx.register_model(model, input_data=features, predict_function="predict_proba")
```

Por que esse contrato:

- **Funciona para qualquer framework sem escrever classe**: o método indicado em `predict_function` vira um modelo `pyfunc` do MLflow, que serializa o objeto com cloudpickle. O cientista não precisa conhecer os flavors do MLflow, FastAPI ou Docker.
- **O schema vem dos dados**: a signature é inferida de `input_data`, então a API valida o input e documenta as features sem nenhuma configuração. As primeiras linhas viram o exemplo do Swagger.
- **A saída `list[list[float]]`** cobre regressão (um valor por instância) e probabilidades (um valor por classe).

Alternativas: o flavor de cada framework (`mlflow.sklearn.log_model`) obrigaria o cientista a conhecer um por framework, e um `PythonModel` próprio dá mais controle (pré e pós-processamento), mas exige escrever uma classe.

Limitações: o input é tabular (DataFrame), a saída precisa ser numérica (um `predict` que retorna labels em string não funciona), o modelo precisa ser serializável com cloudpickle e cada pipeline registra um modelo só, com o nome do pipeline.

## Isolamento

- Cada step roda em seu próprio container, criado da imagem do pipeline, com os limites de `cpus` e `memory` do `@step`.
- O código e a configuração são montados como somente leitura. Dos diretórios do host, só o de dados da execução e o de saída do step podem ser escritos.
- Os containers rodam com o usuário do host, para os arquivos em `.mlbox` não ficarem com o root.

Limitações: os limites do Docker são limites e não reservas, então steps paralelos dividem a mesma máquina e nada impede que a soma passe do que o host tem. Os containers também têm acesso à rede. Uma rede interna que só alcance o MLflow evitaria downloads escondidos durante o treino.

## Reprodutibilidade

O que a ferramenta garante:

- **Ambiente**: uma imagem por pipeline, com tag igual ao hash do `requirements.txt` e do ID da imagem base. Enquanto nada muda a imagem é reaproveitada, e a tag fica registrada na run (`mlbox.image`).
- **Seed**: passada com `--seed` ou gerada, registrada como parâmetro da run, aplicada no `random` e no NumPy antes de cada step e disponível em `ctx.random_seed` para os estimadores. Rodar o exemplo duas vezes com os mesmos parâmetros e seed gera as mesmas métricas.
- **Parâmetros**: registrados na run.
- **Serving**: a API roda na mesma imagem que treinou o modelo, então as dependências são as mesmas.

O que não garante:

- **Versões das dependências**: se o cientista não fixar as versões no `requirements.txt`, um rebuild (por exemplo, depois de mudar a imagem base) pode instalar versões diferentes.
- **Código e dados**: não registro o git SHA do código nem um hash dos dados.
- **Pipeline sem requirements**: a run registra a `mlbox:latest`, que é uma tag que muda.

## Orquestração própria em vez de Prefect

Considerei usar um orquestrador pronto como o Prefect, com cada task rodando um container. Ganharia a UI do DAG e os retries prontos, mas seria mais um serviço rodando e mais uma ferramenta para o cientista entender. Aqui a orquestração faz parte da ferramenta, e o [runner.py](../src/mlbox/runner.py) tem pouco mais de 100 linhas (um thread pool em que cada thread espera um container).

O `Runner` depende só de um protocolo `Executor`, então a troca do Docker por outro backend fica isolada nele. Isso não deixa a ferramenta compatível com o SageMaker no estado atual (ver [O que faria diferente com mais tempo](#o-que-faria-diferente-com-mais-tempo)).

## MLflow

Uso o MLflow para tracking e model registry. O ganho maior é o empacotamento padronizado: o modelo é salvo com a signature, um input de exemplo e as dependências, então a API não tem nada específico do modelo.

Ele também guarda o estado das execuções: o status da run e o status de cada step (como tags). `mlbox list` e `mlbox show` leem de lá. A alternativa era um JSON local por execução, que funcionaria sem o servidor no ar, mas duplicaria informação.

Outras escolhas:

- **A run é criada fora do container**: o `Runner` cria a run e o worker retoma ela pelo ID. Assim qualquer `mlflow.log_*` do cientista cai na run certa sem ele precisar saber disso.
- **Porta 5001 no host**: a 5000 conflita com o AirPlay no macOS.
- **Artefatos**: o modelo vai para o MLflow pelo `register_model`, e outros arquivos o cientista pode logar com `ctx.mlflow.log_artifact`. Os arquivos do `data_path` ficam em `.mlbox/runs/<run_id>/data`, ligados à run pelo ID.

## Serving

- **API própria com FastAPI**: foi para cumprir o requisito de implementar essa parte. Um serving runtime pronto, como o MLServer, funcionaria aqui, e em um cenário real eu priorizaria ele: traz adaptive batching, gRPC, métricas e um contrato padrão.
- **Mesma imagem do treino**: as dependências são as mesmas por construção. O custo é que a imagem de serving carrega tudo do treino, e o FastAPI e o uvicorn vão para todas as imagens de pipeline.
- **Modelo carregado do MLflow no startup**: depende do MLflow estar no ar e pode demorar com modelos grandes. Em produção, um init container que baixa o modelo do S3 para um volume resolveria isso, como faz o KServe.
- **Serve a última versão**: sem `--version`, a versão de maior número. Não há promoção ou aprovação de modelo. Em produção o modelo passaria por uma avaliação antes de ir para o ar, por exemplo com o status de aprovação do Model Registry do SageMaker.
- **Validação delegada ao MLflow**: o schema enforcement do MLflow valida o input e o erro vira HTTP 422. A API só checa as features obrigatórias em todas as instâncias e converte inteiros para float nas features `double` (detalhes em [library.md](library.md#serving-mlbox-serve)). Nulos são aceitos porque o modelo pode tratá-los, como o imputer da regressão logística do exemplo.

## Testes

Não me preocupei tanto em selecionar quais testes criar. Com code agents, escrever teste ficou muito barato. Ainda peço só testes relevantes (está no [AGENTS.md](../AGENTS.md)), mas meu foco maior é a qualidade deles. Agents tendem a criar muitos testes ruins, que verificam se algo foi chamado e como foi chamado em vez do comportamento. Evito isso com regras para os agents e fazendo eles seguirem TDD.

Na prática, os testes usam um MLflow real com sqlite em um diretório temporário e checam o estado gravado, sem mocks. Não há Docker nos testes automatizados: o `Runner` é testado com um executor fake, e o `DockerExecutor`, o build da imagem e o `serve_model` foram validados ponta a ponta com o exemplo do iris.

## Requisitos ambíguos

- **"O cientista fornece somente código Python"**: ele escreve só Python. O `requirements.txt` e os dados são as "dependências e dados" que o próprio enunciado diz que ele fornece. Dockerfile, servidor e rotas ficam com a ferramenta.
- **"Retorna os artefatos e as métricas"**: eles ficam no MLflow (modelo, métricas, parâmetros) e em `.mlbox/runs/<run_id>`. A CLI mostra o run ID e o status, e o resto é visto na UI do MLflow. Não há um comando para baixar os artefatos.
- **"Reprodutível"**: a ferramenta garante o ambiente, os parâmetros e a seed. Um resultado idêntico depende do cientista usar a seed nos estimadores (`random_state=ctx.random_seed`).
- **"Dados"**: a ferramenta não define de onde os dados vêm. Eles podem ser lidos de qualquer lugar no primeiro step, e no exemplo o CSV fica dentro do projeto só por simplicidade.
- **"Como a sua solução funcionaria em produção"**: pode ser o deploy da própria ferramenta na AWS (o `Runner`, os containers, o MLflow e a API rodando em uma VM ou no EKS) ou o que a substituiria. Escolhi a segunda: o SDK do mlbox continua sendo a interface do cientista, mas a orquestração, os jobs, o tracking e o serving passam a ser do SageMaker. Levar a ferramenta como está seria reimplementar o que o SageMaker já faz, e a OLX já usa ele.

## Fora do escopo

- **Escalabilidade**: tudo roda em uma máquina, sem fila nem controle dos recursos disponíveis.
- **Retries, timeout e cache de steps**: um step que falha derruba a execução, e rodar de novo executa todos os steps.
- **Lógica condicional no grafo** (como o `ConditionStep` do SageMaker): decisões ficam no código, como o `evaluate` do exemplo, que escolhe o melhor modelo.
- **Hyperparameter tuning como recurso da plataforma**: fica no código do cientista (um Optuna dentro de um step, por exemplo).
- **GPU e treino distribuído.**
- **Imagens otimizadas**: não há multi-stage build, e a imagem de serving é a de treino, que poderia ser bem mais enxuta.
- **Promoção e aprovação de modelos**: a API serve a última versão.
- **Pré e pós-processamento na API**, input não tabular e saída não numérica.
- **Segurança**: autenticação, multi-tenant, socket do Docker montado no compose, containers com acesso à rede.
- **Monitoramento, drift e inferência em batch.**

## O que faria diferente com mais tempo

- **Compatibilidade de fato com o SageMaker**: hoje a ferramenta só se baseia nele, e o desenho de arquitetura assume uma versão compatível que ainda não existe. Para isso seria preciso seguir os contratos de container à risca (dados em `/opt/ml/input/data`, modelo em `/opt/ml/model`, `/ping` e `/invocations` no serving), trocar o `data_path` local por um prefixo no S3 e traduzir o `Pipeline` do mlbox para o SageMaker Pipelines, com `cpus`/`memory` virando `instance_type`.
- **Reprodutibilidade mais forte**: lock das dependências com hashes, e o git SHA do código e um hash dos dados registrados na run.
- **Cache e retries de steps**: pular um step cujo código, entradas e imagem não mudaram, e retry e timeout configuráveis no `@step`.
- **Imagem de serving enxuta**: uma imagem só com as dependências de inferência (as que o MLflow registra com o modelo, mais o FastAPI), em vez da imagem de treino.

## Tempo gasto

Cerca de 10-12 horas, entre pesquisa, interação com Agent, implementação, testes interativos e documentação.
