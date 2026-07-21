# Geometria dos Embeddings em Recomendação: GNNs e Transformers sob Manifold Learning

Código-fonte e notebook de reprodução do artigo aceito no **XV BraSNAM (Brazilian Workshop on Social Network Analysis and Mining), 2026**.

> **Marinho, A. A.; Pedronette, D. C. G.** *Geometria dos Embeddings em Recomendação: GNNs e Transformers sob Manifold Learning.* XV BraSNAM, 2026. Porto Alegre: SBC.

---

## Visão geral

Comparamos três arquiteturas de recomendação — **LightGCN** (GNN sem-feature), **KGAT** (GNN com grafo de conhecimento e atenção) e **SASRec** (Transformer sequencial) — sob a lente de *manifold learning*, caracterizando como cada modelo molda a geometria dos embeddings aprendidos.

Dois protocolos de avaliação são usados, ambos com *full-corpus ranking*:

- **Predição do próximo item** (cronológica, leave-one-out) — apenas em ML-1M, pois Yelp2018 e Amazon-Book do repositório original não preservam ordem temporal.
- **Recomendação geral** (divisão aleatória 80/20) — nos três datasets, sem SASRec.

Duas métricas geométricas com **grafo de interação como referência**:

- **NP@10** — *Neighborhood Preservation*: fração dos 10 vizinhos no espaço de embedding que também estão entre os 10 vizinhos no grafo de interação (Jaccard).
- **LDD@10** — *Local Distance Distortion*: distorção média das distâncias locais entre o espaço de embedding e o grafo de interação.

A maioria dos experimentos roda com **3 sementes** (2020, 2021, 2022), reportando média ± desvio padrão. Duas exceções, por restrição computacional, estão sinalizadas ao longo do texto: o **Amazon-Book** roda com **semente única (2020)**, e a análise geométrica do **LightGCN no Yelp2018** usa **2 sementes** (a 3ª não foi executada).

### Principais achados

- **SASRec domina** a predição do próximo item em ML-1M-LOO (HR@10 ≈ 0.25 vs ≈ 0.08 do LightGCN).
- **LightGCN iguala ou supera o KGAT** em recomendação geral nos três datasets, **contradizendo a suposição** de que grafos de conhecimento melhoram performance sistematicamente.
- A análise geométrica mostra que o KGAT preserva menos a vizinhança do grafo de interação (NP@10 menor), enquanto o LightGCN mantém melhor essa estrutura original — o que acompanha a vantagem no ranking.

---

## Estrutura do repositório

```
.
├── brasnam_consolidated.ipynb   # Notebook único de reprodução
├── requirements.txt             # Dependências Python
├── README.md                    # Este arquivo
├── LICENSE                      # MIT
└── .gitignore
```

Pastas geradas em runtime (gitignoradas):

```
data/                            # Datasets baixados pelo notebook
brasnam_experiments/             # JSONs de resultados + figuras
└── archive/                     # Backups timestampados (save_results)
```

---

## Como reproduzir

### Opção 1 — Google Colab (recomendado)

1. Abra `brasnam_consolidated.ipynb` no Colab.
2. **Runtime → Change runtime type → GPU** (T4 ou superior).
3. Execute as Seções 1 a 8 (definições — rápido).
4. Execute a **Seção 9** (*smoke test*) — deve terminar em ~10 minutos. Se algo falhar aqui, conserte antes de continuar.
5. Execute a **Seção 10** (experimentos completos). Tempo total estimado em GPU T4 (varia com o hardware alocado):

   | Dataset | Modelos | Tempo aproximado (3 seeds, exceto onde indicado) |
   |---|---|---|
   | MovieLens-1M | LightGCN, KGAT, SASRec | algumas horas |
   | Yelp2018 | LightGCN, KGAT | ~1 dia |
   | Amazon-Book | LightGCN, KGAT | ~22–44h **por seed** (rodado com 1 seed) |

6. Execute a **Seção 11** para consolidar os JSONs do Amazon-Book na tabela final agregada.

### Opção 2 — Local (com GPU NVIDIA)

```bash
git clone https://github.com/Concius/brasnam2026-manifold-recsys
cd brasnam2026-manifold-recsys

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook brasnam_consolidated.ipynb
```

Edite a célula de "Google Drive mount" para usar um diretório local em `SAVE_DIR` (o `try/except` já faz fallback para `./brasnam_experiments`).

### *Resume logic*

Cada cell de experimento checa se o JSON de saída já existe. Se sim, pula. Para forçar re-execução:

```python
FORCE_OVERWRITE = True   # definido na Seção 8
```

---

## Diferenças em relação aos repositórios originais

| Mudança | Motivação |
|---|---|
| **KGAT com atenção TransR** (era Bi-Interaction GCN sem atenção) | A implementação anterior não era KGAT — o mecanismo de atenção do paper (Wang et al. 2019, Eq. 4–5) estava ausente. Cross-validado com [LunaBlack/KGAT-pytorch](https://github.com/LunaBlack/KGAT-pytorch). |
| **SASRec com full-corpus evaluation** (era 1+100 sampled) | A avaliação amostrada (Krichene & Rendle, 2020) é incomparável com LightGCN/KGAT, que avaliam contra todo o catálogo. |
| **SASRec apenas em ML-1M** | Yelp2018 e Amazon-Book do LightGCN repo não preservam ordem cronológica. |
| **Ranking do KGAT avaliado contra todos os usuários** (era amostra não-determinística de 1000) | Torna a avaliação de ranking determinística. (Distinto da amostra de 1000 *itens* usada só nas métricas geométricas NP/LDD — ver §4.2 do artigo.) |
| **NP@10 / LDD@10 com grafo de interação como referência** (era o próprio embedding) | A definição anterior media fidelidade do t-SNE ao embedding, não estrutura intrínseca dos dados. Responde diretamente à crítica dos revisores. |
| **LightGCN também avaliado com NP/LDD** | Faltavam números do LightGCN para esses dois indicadores na versão anterior. |
| **3 seeds + média ± std** (onde computacionalmente viável) | Reprodutibilidade — atende ao revisor 2. Exceções sinalizadas: Amazon-Book (1 seed) e geometria do LightGCN no Yelp2018 (2 seeds). |

---

## Resultados

Resumo das tabelas principais (números completos e desvios no paper):

**Tabela A — Next-item prediction (ML-1M-LOO, full-corpus, 3 seeds)**

| Modelo | HR@10 | Recall@20 | NDCG@10 |
|---|---|---|---|
| LightGCN | 0.084 | 0.149 | 0.041 |
| KGAT | 0.079 | 0.136 | 0.039 |
| **SASRec** | **0.249** | **0.364** | **0.134** |

**Tabela B — Recomendação geral (Yelp2018, full-corpus, 3 seeds)**

| Modelo | HR@10 | Recall@20 | NDCG@10 | NP@10 | LDD@10 |
|---|---|---|---|---|---|
| **LightGCN** | **0.182 ± 0.001** | **0.073 ± 0.000** | **0.037 ± 0.000** | **0.234 ± 0.003** † | 0.740 ± 0.004 † |
| KGAT | 0.161 ± 0.000 | 0.064 ± 0.000 | 0.032 ± 0.000 | 0.068 ± 0.005 | 0.768 ± 0.002 |

† NP/LDD do LightGCN no Yelp2018 são média de **2 seeds** (a análise geométrica da 3ª seed não foi executada). As métricas de ranking usam as 3 seeds.

**Tabela B — Recomendação geral (Amazon-Book, full-corpus, 1 seed: 2020)**

| Modelo | HR@10 | Recall@20 | NDCG@10 | NP@10 | LDD@10 |
|---|---|---|---|---|---|
| **LightGCN** | **0.173** | **0.142** | **0.061** | **0.292** | 0.719 |
| KGAT | 0.145 | 0.124 | 0.049 | 0.186 | 0.709 |

Amazon-Book roda com **semente única (2020)** por custo computacional; completar as seeds é trabalho futuro. Os JSONs com todos os números detalhados são gerados em `brasnam_experiments/` ao rodar o notebook.

---

## Dependências

Versões mínimas testadas (Colab, maio/2026):

- Python ≥ 3.10
- PyTorch ≥ 2.0 (com CUDA)
- NumPy ≥ 1.24, SciPy ≥ 1.10
- scikit-learn ≥ 1.3 (para `KMeans` e `TSNE`)
- pandas ≥ 2.0
- matplotlib ≥ 3.7, seaborn ≥ 0.12

Lista completa em [`requirements.txt`](requirements.txt).

---

## Citação

Se este código for útil para a sua pesquisa, cite:

```bibtex
@inproceedings{marinho2026geometria,
  title     = {Geometria dos Embeddings em Recomenda\c{c}\~ao: GNNs e Transformers sob Manifold Learning},
  author    = {Marinho, Arthur Andreazza and Pedronette, Daniel Carlos Guimar\~aes},
  booktitle = {Anais do XV Brazilian Workshop on Social Network Analysis and Mining (BraSNAM)},
  year      = {2026},
  address   = {Porto Alegre, RS, Brasil},
  publisher = {Sociedade Brasileira de Computa\c{c}\~ao (SBC)},
  doi       = {10.5753/brasnam.2026.22535},
  url       = {https://doi.org/10.5753/brasnam.2026.22535},
}
```

---

## Licença

[MIT](LICENSE) — uso livre com atribuição.
---

## Licença

[MIT](LICENSE) — uso livre com atribuição.
