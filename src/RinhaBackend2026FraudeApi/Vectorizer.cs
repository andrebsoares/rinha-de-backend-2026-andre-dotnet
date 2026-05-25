internal static class Vectorizer
{
    private static float Clamp(float x) => x < 0f ? 0f : x > 1f ? 1f : x;

    internal static float[] Vectorize(FraudScoreRequest req, IReadOnlyDictionary<string, float> mccRisk)
    {
        var tx = req.transaction;
        var customer = req.customer;
        var merchant = req.merchant;
        var terminal = req.terminal;
        var last = req.last_transaction;

        // Timestamps são UTC — usar Hour/DayOfWeek diretamente (sem conversão de timezone)
        var requestedAt = tx.requested_at;

        // [4] day_of_week: seg=0 ... dom=6
        // .NET DayOfWeek: dom=0, seg=1 ... sab=6  →  (value + 6) % 7
        int dotNetDay = (int)requestedAt.DayOfWeek;
        float dayOfWeek = ((dotNetDay + 6) % 7) / 6f;

        // [5] minutes_since_last_tx  |  [6] km_from_last_tx
        float minutesSinceLast;
        float kmFromLast;
        if (last is null)
        {
            minutesSinceLast = -1f;
            kmFromLast = -1f;
        }
        else
        {
            float totalMinutes = (float)(requestedAt - last.timestamp).TotalMinutes;
            minutesSinceLast = Clamp(totalMinutes / 1440f);
            kmFromLast = Clamp(last.km_from_current / 1000f);
        }

        // [11] unknown_merchant
        bool isUnknown = !customer.known_merchants.Contains(merchant.id);

        // [12] mcc_risk — default 0.5 se ausente
        float mccRiskValue = mccRisk.TryGetValue(merchant.mcc, out float risk) ? risk : 0.5f;

        return
        [
            Clamp(tx.amount / 10000f),                                    // [0]  amount
            Clamp(tx.installments / 12f),                                 // [1]  installments
            Clamp((tx.amount / customer.avg_amount) / 10f),               // [2]  amount_vs_avg
            requestedAt.Hour / 23f,                                       // [3]  hour_of_day
            dayOfWeek,                                                     // [4]  day_of_week
            minutesSinceLast,                                              // [5]  minutes_since_last_tx
            kmFromLast,                                                    // [6]  km_from_last_tx
            Clamp(terminal.km_from_home / 1000f),                         // [7]  km_from_home
            Clamp(customer.tx_count_24h / 20f),                           // [8]  tx_count_24h
            terminal.is_online ? 1f : 0f,                                 // [9]  is_online
            terminal.card_present ? 1f : 0f,                              // [10] card_present
            isUnknown ? 1f : 0f,                                          // [11] unknown_merchant
            mccRiskValue,                                                  // [12] mcc_risk
            Clamp(merchant.avg_amount / 10000f),                          // [13] merchant_avg_amount
        ];
    }
}
