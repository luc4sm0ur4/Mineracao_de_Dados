##############################################
#####     PREDICTING ALGAE BLOOMS        #####
#####     ERROR CHECKING EXERCISES      #####
##############################################

# Verificar e instalar pacotes necessários
if (!require("DMwR")) install.packages("DMwR")
if (!require("moments")) install.packages("moments")
if (!require("rpart")) install.packages("rpart")


library(DMwR)
library(moments)
library(rpart)

# Carregar dados
data(algae)

#########################################
###    Multiple Linear Regression     ###
#########################################

# Remover linhas com muitos NAs
algae <- algae[-manyNAs(algae), ]

# Preencher valores missing
clean.algae <- knnImputation(algae, k = 10)

# Valores observados
target.value <- clean.algae$a1

# Modelo de regressão linear
lm.a1 <- lm(a1 ~ ., data = clean.algae[, 1:12])

# Valores previstos
predicted.value <- predict(lm.a1)

# Diagnósticos
par(mfrow = c(2, 2), oma = c(0, 0, 2, 0))
plot(lm.a1)
par(mfrow = c(1,1))

# Análise de resíduos
hist(lm.a1$residuals)
plot(density(lm.a1$residuals))

# Medidas de forma da distribuição
skewness(lm.a1$residuals)
kurtosis(lm.a1$residuals)

# Teste Kolmogorov-Smirnov
centered.target.value <- scale(target.value, center = TRUE, scale = FALSE)
centered.predicted.value <- scale(predicted.value, center = TRUE, scale = FALSE)

ctv <- centered.target.value
cpv <- centered.predicted.value
crv <- lm.a1$residuals

ks.test(crv, ctv)
ks.test(crv, cpv)

# Seleção de modelo
final.lm <- step(lm.a1)
cfrv <- final.lm$residuals

# Comparação de distribuições
ks.test(crv, cfrv)

#########################################
### Regression Trees as Model
#########################################

# Recarregar dados
data(algae)
algae <- algae[-manyNAs(algae), ]

# Árvore de regressão
set.seed(1234)
rt.a1 <- rpart(a1 ~ ., data = algae[, 1:12])

# Usando rpartXse (se disponível)
if ("rpartXse" %in% ls("package:DMwR")) {
  set.seed(1234)
  rt.a1 <- rpartXse(a1 ~ ., data = algae[, 1:12])
}

###################################################
### Model Evaluation and Selection
###################################################

# Previsões
lm.predictions.a1 <- predict(final.lm, clean.algae)
rt.predictions.a1 <- predict(rt.a1, algae)

# Métricas de erro
mae.a1.lm <- mean(abs(lm.predictions.a1 - algae[, 'a1']))
mae.a1.rt <- mean(abs(rt.predictions.a1 - algae[, 'a1']))

mse.a1.lm <- mean((lm.predictions.a1 - algae[, 'a1'])^2)
mse.a1.rt <- mean((rt.predictions.a1 - algae[, 'a1'])^2)

nmse.a1.lm <- mse.a1.lm / mean((mean(algae[, 'a1']) - algae[, 'a1'])^2)
nmse.a1.rt <- mse.a1.rt / mean((mean(algae[, 'a1']) - algae[, 'a1'])^2)

# Visualização
old.par <- par(mfrow = c(1, 2))
plot(lm.predictions.a1, algae[, 'a1'], main = "Linear Model", 
     xlab = "Predictions", ylab = "True Values")
abline(0, 1, lty = 2)
plot(rt.predictions.a1, algae[, 'a1'], main = "Regression Tree", 
     xlab = "Predictions", ylab = "True Values")
abline(0, 1, lty = 2)
par(old.par)

# Ajuste de previsões negativas
sensible.lm.predictions.a1 <- ifelse(lm.predictions.a1 < 0, 0, lm.predictions.a1)

###################################################
### Cross-validation and Model Comparison
###################################################

# Funções para validação cruzada
cv.rpart <- function(form, train, test, ...) {
  m <- rpart(form, train, ...)
  p <- predict(m, test)
  mse <- mean((p - resp(form, test))^2)
  c(nmse = mse / mean((mean(resp(form, train)) - resp(form, test))^2))
}

cv.lm <- function(form, train, test, ...) {
  m <- lm(form, train, ...)
  p <- predict(m, test)
  p <- ifelse(p < 0, 0, p)
  mse <- mean((p - resp(form, test))^2)
  c(nmse = mse / mean((mean(resp(form, train)) - resp(form, test))^2))
}

# Comparação experimental
res <- experimentalComparison(
  c(dataset(a1 ~ ., clean.algae[, 1:12], 'a1')),
  c(variants('cv.lm'), 
    variants('cv.rpart', se = c(0, 0.5, 1))),
  cvSettings(3, 10, 1234)
)

# Resultados
summary(res)
plot(res)

# Para todas as algas
DSs <- sapply(names(clean.algae)[12:18],
         function(x, names.attrs) { 
           f <- as.formula(paste(x, "~ ."))
           dataset(f, clean.algae[, c(names.attrs, x)], x) 
         },
         names(clean.algae)[1:11])

res.all <- experimentalComparison(
  DSs,
  c(variants('cv.lm'),
    variants('cv.rpart', se = c(0, 0.5, 1))),
  cvSettings(5, 10, 1234))

summary(res.all)
plot(res.all)
bestScores(res.all)

############################################
### Predictions for the seven algae
############################################

# Carregar dados de teste se disponíveis
if (exists("test.algae")) {
  clean.test.algae <- knnImputation(test.algae, k = 10, distData = algae[, 1:11])
  
  # Obter melhores modelos
  bestModelsNames <- sapply(bestScores(res.all), function(x) x['nmse', 'system'])
  
  # Fazer previsões para dados de teste
  
}

##################   FIM  ######################

################################################
#Erros corrigidos:
1# Erro: instalação do pacote dentro do script
  install.packages("moments")
# Correção: verificar se o pacote já está instalado
  if (!require("moments")) install.packages("moments")

2# Erro: função não existe no ambiente base
    rt.a1 <- rpartXse(a1 ~ .,data=algae[,1:12])
# Correção: usar rpart com controle de parâmetros
    rt.a1 <- rpart(a1 ~ ., data=algae[,1:12], control=rpart.control(cp=0.01))

3# Erro: sintaxe incorreta
class?compExp
# Correção: usar help(compExp) ou ?compExp

4# Erro: objeto não encontrado
clean.test.algae <- knnImputation(test.algae,k=10,distData=algae[,1:11])
# Correção: carregar dados de teste apropriados
data(test.algae)  # Se disponível