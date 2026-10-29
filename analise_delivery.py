# ==========================================================
# 1. Importações
# ==========================================================
import pandas as pd
import numpy as np

# ==========================================================
# 2. Carregamento dos Dados
# ==========================================================
pedidos = pd.read_csv("dados/pedidos.csv")
cardapio = pd.read_csv("dados/cardapio.csv")

# ==========================================================
# 3. EDA (Análise Exploratória de Dados)
# ==========================================================
print("--- Primeiras 5 linhas ---")
print(pedidos.head())

print("\n--- Últimas 5 linhas ---")
print(pedidos.tail())

print("\n--- Informações Estruturais e Tipos ---")
pedidos.info()

print("\n--- Resumo Estatístico ---")
print(pedidos.describe())

print(f"\nDimensões do DataFrame: {pedidos.shape[0]} linhas e {pedidos.shape[1]} colunas")

# ==========================================================
# 4. Tratamento de Valores Ausentes
# ==========================================================
print("\n--- Valores Ausentes Antes do Tratamento ---")
print(pedidos.isnull().sum())

# Preencher valores ausentes na coluna Quantidade com a média
media_qtd = pedidos["Quantidade"].mean()
pedidos["Quantidade"] = pedidos["Quantidade"].fillna(media_qtd)

# Remover linhas onde Preco_Unitario é nulo
pedidos = pedidos.dropna(subset=["Preco_Unitario"])

# ==========================================================
# 5. Engenharia de Features (Novas Colunas)
# ==========================================================
# Calcular receita de cada item (Quantidade * Preco_Unitario)
pedidos["Receita_Item"] = pedidos["Quantidade"] * pedidos["Preco_Unitario"]

# ==========================================================
# 6. Agregações por Item
# ==========================================================
agrupado_item = pedidos.groupby("Item").agg({
    "Quantidade": "sum",
    "Receita_Item": "sum"
})

print("\n--- Top 5 Itens Mais Vendidos (Quantidade) ---")
print(agrupado_item["Quantidade"].sort_values(ascending=False).head(5))

print("\n--- Top 5 Itens com Maior Receita ---")
print(agrupado_item["Receita_Item"].sort_values(ascending=False).head(5))

# ==========================================================
# 7. Análise Temporal
# ==========================================================
pedidos["Data"] = pd.to_datetime(pedidos["Data"])
pedidos["Mes"] = pedidos["Data"].dt.to_period("M")

receita_mes = pedidos.groupby("Mes")["Receita_Item"].sum()
print("\n--- Receita Total por Mês ---")
print(receita_mes)

# ==========================================================
# 8. Integração de Dados (Merge)
# ==========================================================
df_completo = pd.merge(pedidos, cardapio, on="Item", how="inner")

receita_categoria = df_completo.groupby("Categoria")["Receita_Item"].sum().sort_values(ascending=False)
print("\n--- Receita por Categoria ---")
print(receita_categoria)
print(f"\nCategoria que mais fatura: {receita_categoria.idxmax()}")

# ==========================================================
# 9. Filtros e Consultas
# ==========================================================
filtro_salgados = df_completo[(df_completo["Categoria"] == "Salgados") & (df_completo["Quantidade"] > 10)]
print("\n--- Pedidos de Salgados com Mais de 10 Unidades ---")
print(filtro_salgados.head())

# ==========================================================
# 10. KPIs com NumPy
# ==========================================================
receita_total = np.sum(pedidos["Receita_Item"].values)
total_itens = np.sum(pedidos["Quantidade"].values)
ticket_medio = receita_total / len(pedidos)

print("\n--- Indicadores Principais de Desempenho (KPIs) ---")
print(f"Receita Total: R$ {receita_total:,.2f}")
print(f"Total de Itens Vendidos: {total_itens:,.0f}")
print(f"Ticket Médio: R$ {ticket_medio:,.2f}")
