# ⚽ World Cup Analyst AI

An intelligent match prediction and historical analysis chatbot powered by **LangChain**, **FAISS**, and **OpenAI**, covering 90+ years of FIFA World Cup history (1930–2022).

---

## 🚀 What It Does

The system combines a **RAG (Retrieval-Augmented Generation)** pipeline with a **LangChain agentic framework** to answer questions about World Cup history, generate match reports, predict outcomes, and surface player statistics — all grounded in verified historical data.

It uses **7 pipeline tools** ranging from semantic vector search to raw Pandas computations, coordinated by a `gpt-3.5-turbo` agent. A **FastAPI backend** serves the agent via an ngrok tunnel, and a **Vercel frontend** provides the chat interface.

---

## 🏗️ Architecture

```
User → Vercel Frontend → ngrok → FastAPI Backend → LangChain Agent
                                                        ↓
                                          7 Tools (FAISS, Pandas, ELO, etc.)
                                                        ↓
                                     Google Drive CSVs + Local FAISS Index
```

**Core Stack:**

* **Orchestration:** LangChain `create\_tool\_calling\_agent` + `AgentExecutor`
* **LLM:** OpenAI `gpt-3.5-turbo`
* **Vector Store:** FAISS (1,047 embedded match \& team summary chunks)
* **Backend:** FastAPI + pyngrok (tunneled from Google Colab)
* **Frontend:** Vercel

---

## 📂 Dataset Setup (Required Before Running)

The notebook reads all data from a folder called `WorldCup/` in your **Google Drive root**. You must create this folder and upload the following 8 CSV files before running the notebook.

### Required Files

| File | Description | Source |
|------|-------------|--------|
| `FIFA World Cup 19302022 All Match Dataset.csv` | Core match records (964 rows) | Kaggle |
| `results.csv` | International results 1872–present | Kaggle |
| `teams\_match\_features.csv` | ELO scores + FIFA player ratings per match | Kaggle |
| `teams\_form.csv` | Rolling form metrics (win rate, avg goals) | Kaggle |
| `goalscorers.csv` | Goal-level data with scorer, minute, penalty flag | Kaggle |
| `shootouts.csv` | Penalty shootout results | Kaggle |
| `player\_aggregates.csv` | Aggregated player-level WC stats | Kaggle |
| `former\_names.csv` | 36 historical country name mappings (e.g. West Germany → Germany) | Kaggle |

### How to Upload

1. Go to [drive.google.com](https://drive.google.com)
2. Create a folder named exactly `WorldCup` in **My Drive**
3. Upload all 8 CSV files into that folder
4. The notebook will mount your Drive and read from `/content/drive/MyDrive/WorldCup/`

> ⚠️ The `FIFA World Cup 19302022 All Match Dataset.csv` file uses \*\*CP-1252 encoding\*\* — the notebook handles this automatically.

---

## 🔧 Running the Notebook (Google Colab)

### Step 1 — Cell 1A: Dependency Installation \& Why It Restarts

Cell 1A installs a set of **version-pinned libraries** (e.g. `numpy==1.26.4`, `langchain==0.3.7`, `faiss-cpu==1.9.0`). These specific versions are required for compatibility between LangChain, OpenAI, and FAISS.

Google Colab comes pre-loaded with newer versions of many of these packages. Simply `pip install`-ing on top of them is not enough — Python's import system has already loaded the old versions into memory. The only reliable way to apply the new versions is to **fully restart the Python runtime**, which is why Cell 1A ends with:

```python
import os; os.kill(os.getpid(), 9)
```

This forcibly kills the current runtime process, triggering Colab's automatic restart. You will see a "Runtime restarted" message — this is expected. **Do not re-run Cell 1A after the restart.** Proceed directly to Cell 1B and run all subsequent cells in order.

### Step 2 — Run All Remaining Cells in Order

| Cell | What It Does |
|------|-------------|
| \*\*1B\*\* | Imports, API key setup, LLM connection test |
| \*\*2\*\* | Loads \& cleans all 8 CSVs, normalises team names, parses dates |
| \*\*3\*\* | Builds FAISS vector index from match records (saves to Drive) |
| \*\*4\*\* | Prediction engine — H2H, form, ELO, shootout stats |
| \*\*5\*\* | LangChain agent with 6 tools (local smoke test) |
| \*\*6A\*\* | FastAPI backend with 7 tools + chart endpoints |
| \*\*6B\*\* | Opens ngrok tunnel and prints the public API URL |

Paste the ngrok URL into your Vercel environment variable (`NEXT\_PUBLIC\_API\_URL` or equivalent) to connect the frontend.

> ℹ️ The FAISS index is built once in Cell 3 and saved to `WorldCup/faiss\_index/` in your Drive. On subsequent sessions, it loads from cache — no re-embedding needed.

---

## 🔑 Colab Secrets Required

Add these in **Colab → Secrets (🔑)**:

* `OpenAI\_API\_Key` — your OpenAI API key (`sk-...`)
* `Ngrok` — your ngrok auth token

---

## 🛠️ LangChain Pipeline Tools

The agent is equipped with 7 custom tools, each mapped to a distinct stage of the analytical pipeline. The LangChain agent decides which tool(s) to call based on the user's query — often chaining multiple tools together for complex questions like match predictions.

---

### 1\. `dataset\_discovery\_tool` — Pipeline Stage 1

**When it's called:** User asks what data or teams are available.

Returns a metadata overview of all loaded datasets — total match counts, date ranges, team list, and record counts for goalscorers and shootouts. Acts as the agent's awareness layer, letting it know the scope of available data before diving into specifics.

---

### 2\. `data\_ingestion\_tool` — Pipeline Stage 2

**When it's called:** Any question about a specific team's World Cup record.

Computes a verified, real-time summary for a given team directly from the raw CSVs — matches played, wins, draws, losses, goals scored and conceded, goal difference, shootout wins, and recent form metrics. The agent is instructed to **always** call this tool before stating any team statistic, preventing hallucination of made-up numbers.

---

### 3\. `retrieval\_or\_filter\_tool` — Pipeline Stage 3

**When it's called:** Historical questions — specific matches, tournament results, upsets, finals, or any general World Cup history query.

Performs **semantic search** over the FAISS vector store containing 1,047 embedded document chunks (964 match records + 83 team summaries). Returns the most relevant chunks for the query. This is the **primary fallback** for conversational and open-ended historical questions that don't require computation.

---

### 4\. `reasoning\_or\_aggregation\_tool` — Pipeline Stage 4

**When it's called:** Every prediction or team comparison request — no exceptions.

This is the core prediction engine. For any two teams, it aggregates all available signals into one structured payload:

* Head-to-head record (all competitions + World Cup only)
* Most recent 5 matches between the teams
* Recent form (win rate, avg goals scored/conceded)
* ELO ratings and ELO differential
* FIFA player ratings (overall, attack, defense)
* Penalty shootout history

The agent uses this structured data as the factual foundation before forming any prediction or opinion.

---

### 5\. `llm\_synthesis\_tool` — Pipeline Stage 5

**When it's called:** Questions about top scorers, key players, or goal-level data.

Queries the pre-filtered `WC\_GOALS\_ONLY` dataset (World Cup goals only, own goals excluded) to surface top scorers for a specific team or all-time. Also breaks down penalty goals separately. Pass `"all"` as the team name to get the all-time World Cup top scorer leaderboard.

---

### 6\. `report\_generation\_tool` — Pipeline Stage 6

**When it's called:** User asks for a full structured match preview or report.

Generates a formatted, human-readable match preview report combining all prediction signals — H2H record, recent form, ELO ratings, FIFA player ratings, and shootout history — with a data limitations disclaimer appended at the end. Designed to be the single-call output for "give me a full report on X vs Y" queries.

---

### 7\. `analytics\_tool` — Pipeline Stage 7

**When it's called:** Advanced statistical queries that require direct computation on raw data.

Handles queries that go beyond retrieval or standard stats, including:

* **Hat-tricks** — lists all WC hat-tricks, with fastest hat-trick detection by minute span
* **Highest scoring matches** — top 5 by total goals
* **Biggest wins** — top 5 by goal margin
* **Most appearances** — teams with the most WC matches played
* **Shootout records** — teams with most penalty shootout wins
* **Own goals** — all recorded own goals in WC history
* **Tournament statistics** — total goals and goals-per-game by World Cup edition

All computations run directly on Pandas DataFrames from the loaded CSVs, ensuring accuracy without relying on the vector store.

---

### How the Agent Chooses Tools

The agent uses `create\_tool\_calling\_agent` with `gpt-3.5-turbo`. Each tool has a detailed docstring that acts as its routing description. The agent reads these descriptions and selects the appropriate tool(s) for each query. For complex requests (e.g. "predict Brazil vs Germany"), it typically chains **Stage 4 → Stage 2 → Stage 6** in sequence before composing a final answer.

A `ConversationBufferMemory` maintains session context, and baseline instructions are pre-injected into memory at startup to enforce data-driven, unbiased behaviour across the entire conversation.

---

## ⚠️ Limitations

* Data covers World Cup history **up to 2022** only
* Does not reflect post-2022 squad changes, injuries, or managerial shifts
* ELO/FIFA ratings are proxies, not guarantees — predictions are statistical estimates
* The ngrok tunnel is only active while the Colab session is running

---

## 👥 Team

| Member | Role |
|--------|------|
| \*\*Naveen\*\* | LangChain agent architecture, Colab Notebook creation, FastAPI backend, Vercel deployment |
| \*\*Gargi\*\* | Data engineering, Dataset sourcing & Cleaning, team name normalisation pipeline, ELO/FIFA feature merging |
| \*\*Venkat\*\* | Project documentation, presentation design, narrative structure, Testing |

> For educational purposes only. Not professional sports analytics or betting advice.

