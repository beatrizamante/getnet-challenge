# Getnet — Multi-Agent Support System

Sistema de suporte multi-agente para a Getnet, construído com LangGraph, pydantic-ai, FastAPI, ChromaDB e Redis.

---

## Arquitetura

A aplicação segue **Clean Architecture** com separação estrita por camadas. Dependências apontam sempre para dentro (domain → application → infrastructure/interface).

```
src/
├── domain/               # Regras de negócio puras — sem dependências externas
│   ├── entities/         # Modelos pydantic validados (UserMessage, Transaction, Chunk, etc.)
│   ├── ports/            # Interfaces abstratas (ABC) para dependências externas
│   └── shared/           # Erros customizados (AppError e subclasses)
├── application/          # Casos de uso, orquestração de agentes
│   ├── rag_pipeline/     # Ingestão → embedding → ChromaDB → recuperação
│   └── caching/          # Lógica de cache sobre CachePort
├── infrastructure/       # Implementações concretas dos ports
│   ├── adapters/         # ChromaDB, Redis, LLM, Search (implementam os ports)
│   ├── database/         # Clientes de banco de dados
│   └── queues/           # ARQ workers e configuração de filas
├── interface/            # Entradas da aplicação
│   ├── http/             # FastAPI — rotas e server
│   ├── SSE/              # Server-Sent Events para streaming de respostas
│   └── workers.py        # Entry point dos ARQ workers
├── config/               # Configurações e logger
└── _lib/
    └── container.py      # Container de injeção de dependência
```

---

## Agentes

| Agente | Responsabilidade |
|---|---|
| **Router Agent** | Entry point. Analisa a mensagem e decide qual agente especializado acionar (`RouteDecision`) |
| **Knowledge Agent** | RAG sobre o site da Getnet + web search para perguntas gerais sobre produtos/serviços |
| **Customer Support Agent** | Recupera dados do usuário (perfil, transações) para resolver suporte contextualizado |

O grafo de agentes é orquestrado via **LangGraph**. O estado que flui pelo grafo é `ConversationState`.

---

## Stack

| Lib | Uso |
|---|---|
| `langgraph` | Orquestração do grafo de agentes |
| `pydantic-ai` | Definição e execução dos agentes LLM |
| `pydantic` | Validação de todas as entidades de domínio |
| `fastapi` | API HTTP — recebe `{ message, user_id }` e retorna `AgentResponse` |
| `chromadb` | Vector store para o pipeline RAG |
| `redis` + `arq` | Cache de respostas e fila de tarefas assíncronas |
| `deepeval` | Avaliação de qualidade das respostas LLM |

---

## Ports (contratos de domínio)

Cada dependência externa tem um port abstrato em `domain/ports/`. Os adapters em `infrastructure/adapters/` implementam esses ports e são injetados via `_lib/container.py`.

| Port | Adapter esperado |
|---|---|
| `CachePort` | Redis |
| `VectorStorePort` | ChromaDB |
| `EmbeddingPort` | OpenAI / outro provider |
| `LLMPort` | OpenAI / outro provider |
| `SearchPort` | Tavily / SerpAPI |
| `UserRepositoryPort` | Mock / banco de dados |

---

## Como rodar

```bash
# Instalar dependências
uv sync

# Rodar a aplicação
uv run python -m src.main

# Rodar com Docker
docker compose up --build
```

---

## Variáveis de ambiente

> Documentar aqui as env vars necessárias (API keys, URLs de serviços, etc.) quando forem definidas.

---

## Testes

> Estratégia de testes a documentar conforme implementação avança.
