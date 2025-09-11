################################################
#####   Detecting Faudulent Transactions   #####
################################################

# Verificar e instalar pacotes necessários
if (!require("DMwR")) install.packages("DMwR")

library(DMwR)

# Carregar dados
data(sales)

##########################################
######    Exploring the data set    ######
##########################################

# Estatísticas descritivas
summary(sales)

# Número de níveis
nlevels(sales$ID)
nlevels(sales$Prod)

# Verificar valores missing
sum(is.na(sales$Quant) & is.na(sales$Val))

# Proporção de fraudes
table(sales$Insp) / nrow(sales) * 100

# Transações por vendedor e produto
totS <- table(sales$ID)
totP <- table(sales$Prod)

# Preço unitário
sales$Uprice <- sales$Val / sales$Quant
summary(sales$Uprice)

# Produtos mais caros e mais baratos
attach(sales)
upp <- aggregate(Uprice, list(Prod), median, na.rm = TRUE)

topP <- sapply(c(T, F), function(o) 
  upp[order(upp[, 2], decreasing = o)[1:5], 1])
colnames(topP) <- c('Expensive', 'Cheap')
topP

# Análise de vendas por vendedor
vs <- aggregate(Val, list(ID), sum, na.rm = TRUE)
scoresSs <- sapply(c(T, F), function(o) 
  vs[order(vs$x, decreasing = o)[1:5], 1])
colnames(scoresSs) <- c('Most', 'Least')
scoresSs

# Análise de quantidade por produto
qs <- aggregate(Quant, list(Prod), sum, na.rm = TRUE)
scoresPs <- sapply(c(T, F), function(o) 
  qs[order(qs$x, decreasing = o)[1:5], 1])
colnames(scoresPs) <- c('Most', 'Least')
scoresPs

# Identificar outliers
out <- tapply(Uprice, list(Prod = Prod),
              function(x) length(boxplot.stats(x)$out))

sum(out)
sum(out) / nrow(sales) * 100

###################################
######     Data problems     ######
###################################

# Remover transações com ambos Val e Quant missing
detach(sales)
sales <- sales[-which(is.na(sales$Quant) & is.na(sales$Val)), ]

# Remover produtos problemáticos
sales <- sales[!sales$Prod %in% c('p2442', 'p2443'), ]
sales$Prod <- factor(sales$Prod)

# Preencher valores missing
tPrice <- tapply(sales[sales$Insp != 'fraud', 'Uprice'],
                 list(sales[sales$Insp != 'fraud', 'Prod']), median, na.rm = TRUE)

noQuant <- which(is.na(sales$Quant))
sales[noQuant, 'Quant'] <- ceiling(sales[noQuant, 'Val'] / tPrice[sales[noQuant, 'Prod']])

noVal <- which(is.na(sales$Val))
sales[noVal, 'Val'] <- sales[noVal, 'Quant'] * tPrice[sales[noVal, 'Prod']]

# Recalcular preço unitário
sales$Uprice <- sales$Val / sales$Quant

# Salvar dados limpos
save(sales, file = 'salesClean.Rdata')

#######################################################
######     Few Transactions of Some Products     ######
#######################################################

# Análise de produtos com poucas transações
attach(sales)
notF <- which(Insp != 'fraud')
ms <- tapply(Uprice[notF], list(Prod = Prod[notF]), function(x) {
  bp <- boxplot.stats(x)$stats
  c(median = bp[3], iqr = bp[4] - bp[2])
})

ms <- matrix(unlist(ms), length(ms), 2, byrow = TRUE,
             dimnames = list(names(ms), c('median', 'iqr')))

# Visualização
par(mfrow = c(1, 2))
plot(ms[, 1], ms[, 2], xlab = 'Median', ylab = 'IQR', main = '')
plot(ms[, 1], ms[, 2], xlab = 'Median', ylab = 'IQR', main = 'Distributions of Unit Prices',
     col = 'grey', log = "xy")

smalls <- which(table(Prod) < 20)
points(log(ms[smalls, 1]), log(ms[smalls, 2]), pch = '+')

# Análise de similaridade entre produtos
dms <- scale(ms)
smalls <- which(table(Prod) < 20)
prods <- tapply(sales$Uprice, sales$Prod, list)

similar <- matrix(NA, length(smalls), 7, 
                  dimnames = list(names(smalls),
                                  c('Simil', 'ks.stat', 'ks.p', 'medP', 'iqrP', 'medS', 'iqrS')))

for(i in seq(along = smalls)) {
  d <- scale(dms, dms[smalls[i], ], FALSE)
  d <- sqrt(drop(d^2 %*% rep(1, ncol(d))))
  stat <- ks.test(prods[[smalls[i]]], prods[[order(d)[2]]])
  similar[i, ] <- c(order(d)[2], stat$statistic, stat$p.value, ms[smalls[i], ], ms[order(d)[2], ])
}

# Salvar resultados
save(similar, file = 'similarProducts.Rdata')
#############################################

#Erros corrigidos:

1# Erro: possível divisão por zero
sales$Uprice <- sales$Val/sales$Quant
# Correção: adicionar verificação
sales$Uprice <- ifelse(sales$Quant == 0, NA, sales$Val/sales$Quant)

2# Erro: NA em cálculos de agregação
upp <- aggregate(Uprice,list(Prod),median,na.rm=T)
# Correção: garantir que não há NA restantes
upp <- aggregate(Uprice,list(Prod),median,na.rm=T)

3# Erro: níveis de fator não atualizados
sales$Prod <- factor(sales$Prod)
# Correção: usar droplevels()
sales$Prod <- droplevels(sales$Prod)

4# Erro: loop lento para grandes conjuntos de dados
for(i in 1:nrow(clean.test.algae)) 
  preds[i,] <- sapply(1:7, function(x) 
    predict(bestModels[[x]],clean.test.algae[i,]))
# Correção: usar vectorização ou apply
preds <- sapply(bestModels, predict, newdata=clean.test.algae)
############################################