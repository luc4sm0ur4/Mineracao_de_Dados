#Lucas Carvalho da Luz Moura
#Matrícula: 2020111816

#Codigo para exportar o dado 'sales' do R para um arquivo CSV
#para que o Python possa ler esse arquivo CSV

# Carrega o workspace salvo
load("Local_do_arquivo_em_Rdata") # aqui deve ser o local do arquivo .Rdata

# O script de fraude (Fraudulent-Transactions-2.R) depende do objeto 'sales'.
# Vamos salvar esse objeto 'sales' como um arquivo CSV.
# row.names=FALSE é importante para que o pandas (Python) leia corretamente.
write.csv(sales, "Local_que_deseja_salvar", row.names = FALSE) # aqui deve ser o local onde deseja salvar o arquivo .csv

print("Arquivo 'sales_data.csv' exportado com sucesso!")