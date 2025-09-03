# Pacotes necessários (caso não tenha)
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(readr, dplyr, ggplot2, magrittr, plotly, scales, tidyr, lubridate, viridis)

# 1. CARREGAR E PREPARAR OS DADOS -----------------------------------------------
# Dados do Censo Escolar
tb_censo_escolar <- read_delim(
  "https://raw.githubusercontent.com/allanvc/book_IADR-T/master/datasets/tab_censo_sample_ES_MS_RR.csv",
  delim = ";",
  locale = locale(decimal_mark = ",", grouping_mark = ".")
)

# Dados do IBAMA
tb_ibama <- read_csv(
  "https://raw.githubusercontent.com/allanvc/book_IADR-T/master/datasets/PA%20GF%202017%20jan-jun_editada.csv",
  col_types = cols(
    X1 = col_double(),
    TIPO_GF = col_character(),
    STATUS_GF = col_character(),
    UF_REMETENTE = col_character(),
    MUNICÍPIO_REMETENTE = col_character(),
    TIPO_DESTINO = col_character(),
    CEPROF_DESTINATÁRIO = col_character(),
    UF_DESTINATÁRIO = col_character(),
    MUNICÍPIO_DESTINATÁRIO = col_character(),
    N_AUTORIZAÇÃO = col_character(),
    PROCESSO = col_character(),
    EMISSAO = col_integer(),
    NOME_CIENTÍFICO = col_character(),
    PRODUTO = col_character(),
    VOLUME = col_double(),
    UNID = col_character(),
    PRECO_TOTAL = col_double()
  )
)

# Preparação dos dados do IBAMA
tb_ibama$STATUS_GF[1:50000] <- rep("NÃO VERIFICADO", 50000)
tb_ibama2 <- tb_ibama %>%
  mutate(
    preco_unidade = ifelse(VOLUME > 0, PRECO_TOTAL / VOLUME, NA),
    preco_unidade_vezes_1000 = preco_unidade * 1000,
    volume_categoria = cut(VOLUME, 
                          breaks = c(0, 10, 50, 100, 500, 1000, Inf),
                          labels = c("0-10", "10-50", "50-100", "100-500", "500-1000", "1000+"))
  )

# Filtragem de dados problemáticos que podem ter influenciado a análise
tb_ibama_filtered <- tb_ibama2 %>%
  filter(VOLUME > 0, PRECO_TOTAL > 0, is.finite(preco_unidade))

# 2. GRÁFICOS PRINCIPAIS ------------------------------------------------------

# Gráfico 1: Geração de Dispersão com linha de tendência
g1 <- ggplot(tb_ibama_filtered, aes(x = VOLUME, y = PRECO_TOTAL)) +
  geom_point(alpha = 0.6, aes(color = TIPO_GF)) +
  geom_smooth(method = "lm", se = FALSE, color = "darkred") +
  scale_x_continuous(labels = comma, 
                     limits = c(0, quantile(tb_ibama_filtered$VOLUME, 0.95, na.rm = TRUE))) +
  scale_y_continuous(labels = comma, 
                     limits = c(0, quantile(tb_ibama_filtered$PRECO_TOTAL, 0.95, na.rm = TRUE))) +
  labs(title = "Relação entre Volume e Preço Total",
       x = "Volume", y = "Preço Total", color = "Tipo de GF") +
  theme_minimal() +
  theme(legend.position = "bottom")

# Gráfico 2: Geração Boxplot do preço por unidade
g2 <- ggplot(tb_ibama_filtered, aes(x = TIPO_GF, y = preco_unidade)) +
  geom_boxplot(aes(fill = TIPO_GF), outlier.shape = NA) +
  scale_y_continuous(labels = comma, 
                     limits = quantile(tb_ibama_filtered$preco_unidade, c(0.05, 0.95), na.rm = TRUE)) +
  labs(title = "Distribuição do Preço por Unidade por Tipo de GF",
       x = "Tipo de GF", y = "Preço por Unidade") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none")

# Gráfico 3: Volume por estado e tipo de GF
g3 <- tb_ibama2 %>%
  group_by(UF_DESTINATÁRIO, TIPO_GF) %>%
  summarise(Volume_Total = sum(VOLUME, na.rm = TRUE), .groups = 'drop') %>%
  ggplot(aes(x = UF_DESTINATÁRIO, y = Volume_Total, fill = TIPO_GF)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_y_continuous(labels = comma) +
  labs(title = "Volume Total por Estado Destino e Tipo de GF",
       x = "Estado Destino", y = "Volume Total", fill = "Tipo de GF") +
  theme_minimal()

# 3. GRÁFICO MELHORADO COM FACETTING ------------------------------------------

# Ajuste de limites para cada faceta individualmente
g1_melhorado <- ggplot(tb_ibama_filtered, aes(x = VOLUME, y = PRECO_TOTAL)) +
  geom_point(alpha = 0.6, aes(color = TIPO_GF, size = preco_unidade)) +
  geom_smooth(method = "lm", se = FALSE, color = "darkred") +
  facet_wrap(~ UF_DESTINATÁRIO, scales = "free") +
  scale_x_continuous(labels = comma) +
  scale_y_continuous(labels = comma) +
  scale_size_continuous(
    name = "Preço por Unidade",
    breaks = c(100, 1000, 10000),
    labels = comma,
    range = c(1, 5)
  ) +
  scale_color_viridis_d() +
  labs(
    title = "Relação entre Volume e Preço Total por Estado Destino",
    subtitle = "Tamanho do ponto proporcional ao preço por unidade",
    x = "Volume", y = "Preço Total", color = "Tipo de GF"
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    plot.title = element_text(hjust = 0.5, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5),
    strip.background = element_rect(fill = "lightgray"),
    panel.grid.minor = element_blank()
  )

# 4. FUNÇÃO PARA VISUALIZAR OS GRÁFICOS DE FORMA SIMPLIFICADA ---------------------------------------------------
print(g1)
print(g2)
print(g3)
print(g1_melhorado)

# 5. GERAÇÃO DE GRÁFICOS INTERATIVOS COM PLOTLY ------------------------------------------
g1_interactive <- ggplotly(g1)
g2_interactive <- ggplotly(g2)
g3_interactive <- ggplotly(g3)
g1_melhorado_interactive <- ggplotly(g1_melhorado)

# Função para exibir gráficos interativos
g1_interactive
g2_interactive
g3_interactive
g1_melhorado_interactive

# Salvar gráficos em arquivos
ggsave("grafico1.png", g1, width = 10, height = 6)
ggsave("grafico2.png", g2, width = 10, height = 6)
ggsave("grafico3.png", g3, width = 10, height = 6)
ggsave("grafico_melhorado.png", g1_melhorado, width = 12, height = 8)