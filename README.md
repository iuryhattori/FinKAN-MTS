## Decisões de Pré-processamento

### Por que não utilizamos `fill_time_gaps`

O pipeline **não realiza upsample temporal** para preencher slots de 15 minutos
ausentes fora do horário de pregão. Essa decisão foi tomada pelos seguintes motivos:

1. **A B3 opera apenas de segunda a sexta, das 10h às 17h55.** Gerar timestamps
   para períodos fora desse horário (noites, fins de semana, feriados) criaria
   dados sintéticos sem respaldo em negociações reais.

2. **Interpolação sobre gaps longos corrompe o sinal.** Uma rampa linear entre
   a sexta 17h45 e a segunda 10h00 (~65 horas) não representa nenhum fenômeno
   de mercado — o mercado simplesmente não existiu nesse período.

3. **Modelos de séries temporais operam em tempo de pregão, não calendário.**
   Sexta 17h45 → segunda 10h00 são candles consecutivos do ponto de vista do
   modelo, assim como qualquer outro par de candles adjacentes.