# Mensagem para o orientador

Prezado professor,

Sobre o ponto que o senhor levantou, o da "comparação informal" entre
representações anonimizadas e não anonimizadas: o senhor tinha razão, e realizei
o experimento sistematizado antes da defesa.

## O que foi feito

Rodei uma **ablação pareada** sobre um dos sistemas industriais do estudo
original, em versão mais recente. O desenho aproveita uma propriedade do método
que torna a comparação bastante limpa: a anonimização é um mapeamento
determinístico por caractere aplicado no momento da renderização, ou seja, altera
as intensidades dos pixels e não mexe na geometria. O mesmo arquivo pode então
ser renderizado duas vezes, gerando duas imagens de dimensões, layout e rótulo
idênticos, diferindo apenas no alfabeto de intensidades. As duas versões são
produzidas na mesma passada, o que garante o pareamento instância a instância e
permite usar testes pareados, bem mais sensíveis do que comparar execuções
independentes.

Configuração: 12 classes de estereótipo (inferidas por sufixo de nome de classe e
por pacote), 10.923 pares de minimapas, descritores LBP, Random Forest,
validação cruzada estratificada 10×5 com os mesmos splits para as duas
representações.

**Resultado:**

| | |
|---|---|
| Acurácia sem anonimização | 0,8015 ± 0,0069 |
| Acurácia com anonimização | 0,8076 ± 0,0063 |
| Diferença (custo de anonimizar) | −0,0060 |
| IC 95% | [−0,0123, +0,0003] |
| Teste de diferença | p = 0,060 |
| Equivalência dentro de ±2pp (TOST) | p = 2,4 × 10⁻⁵ |

A anonimização não custou desempenho — na verdade ficou 0,6 ponto percentual
acima. O limite superior do intervalo de confiança coloca a maior perda
compatível com os dados em 0,0003, ou seja, praticamente zero.

Uma observação metodológica que vale registrar. O senhor sugeriu verificar se há
diferença estatisticamente significativa, e acabei usando dois testes, porque
são perguntas diferentes. O teste de diferença dá p = 0,060, não significativo —
mas isso sozinho permitiria apenas dizer "não detectei diferença", o que é frágil,
já que poderia refletir só falta de poder. O que sustenta a proposta é o teste de
equivalência (TOST), com margem de 2 pontos percentuais fixada antes da execução:
ele mostra que a diferença está demonstradamente dentro da margem. O McNemar
ainda detecta diferença em 6 das 10 repetições, e a direção favorece a versão
anonimizada.

Complementei com uma medição da própria transformação, que não exige treinar
modelo nenhum: a anonimização reduz o alfabeto de 152 para 37 níveis de
intensidade e a entropia por pixel de 5,444 para 2,358 bits, uma **redução de
56,7%**. Apenas 25,9% dos caracteres sobrevivem intactos, e são exatamente os
símbolos: pontuação, delimitadores e operadores.

Juntando as duas medidas, chega-se ao resultado que considero mais forte:
**descarta-se mais da metade da informação por pixel e a acurácia não cai.** É
evidência direta de que o sinal discriminativo é estrutural, não lexical, que é
a tese central do trabalho.

## O que ficou por fazer

Registrei os limites do resultado no texto, como escopo da atividade A2 e não
como generalização:

1. **Um sistema apenas.** Estabelece que a anonimização *pode* ser gratuita, não
   que sempre é. A replicação no corpus de 100 projetos open source está no
   cronograma para julho–agosto, e lá o dataset também poderá ser publicado para
   replicação externa.

2. **Um descritor e uma família de classificadores.** LBP descreve relações
   locais de intensidade e pode ser constitucionalmente cego justamente ao que a
   anonimização remove. Um modelo convolucional, que aprende os próprios filtros,
   poderia explorar diferenças lexicais que o LBP ignora — então o resultado não
   se transfere automaticamente para o estudo em larga escala. Essa repetição
   está prevista para o período do sanduíche.

3. **Apenas o objetivo Tipo.** Projeto e Autor ainda precisam ser avaliados, e
   Autor em particular é o que mais depende de estilo, portanto o mais provável
   de ser sensível à supressão.

4. **A curva graduada** de níveis de supressão que a A2 promete continua
   pendente; o que fiz foi a comparação de dois níveis, ligado e desligado.

Os scripts ficaram versionados no repositório, então essas extensões são questão
de apontar para outros dados, não de reimplementar nada.

## Demais ajustes

Também apliquei os outros pontos da sua revisão: o termo "artesanais" no resumo,
o alinhamento justificado, a padronização de "Gebara Jr." nas referências, a
redução dos travessões, os parágrafos introdutórios dos capítulos, a explicação
do operador LBP da Figura 5 e a regeneração da Figura 6 com eixos legíveis.

Dois deles renderam correções de conteúdo que vale mencionar. Na Figura 6,
descobri que a versão anterior cortava o eixo vertical e escondia que dois
códigos concentram entre 91% e 99% da massa do descritor; a nova versão mostra o
descritor completo em escala logarítmica e a região discriminativa separadamente.
E na Tabela 7, as três colunas de precision, recall e F1 traziam o mesmo número
repetido — o artigo do IWSSIP reporta apenas acurácia, e essas métricas nunca
foram calculadas para aquele experimento. A tabela agora reporta acurácia em
coluna única.

Fico à disposição.
