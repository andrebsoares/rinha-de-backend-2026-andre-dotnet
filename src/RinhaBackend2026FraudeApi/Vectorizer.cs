internal static class Vectorizer
{
    private static float Clamp(float x) => x < 0f ? 0f : x > 1f ? 1f : x;

    internal static void Vectorize(FraudScoreRequest req, IReadOnlyDictionary<string, float> mccRisk, Span<float> result)
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

        result[0] = Clamp(tx.amount / 10000f);                          // amount
        result[1] = Clamp(tx.installments / 12f);                       // installments
        result[2] = Clamp((tx.amount / customer.avg_amount) / 10f);     // amount_vs_avg
        result[3] = requestedAt.Hour / 23f;                             // hour_of_day
        result[4] = dayOfWeek;                                           // day_of_week
        result[5] = minutesSinceLast;                                    // minutes_since_last_tx
        result[6] = kmFromLast;                                          // km_from_last_tx
        result[7] = Clamp(terminal.km_from_home / 1000f);               // km_from_home
        result[8] = Clamp(customer.tx_count_24h / 20f);                 // tx_count_24h
        result[9] = terminal.is_online ? 1f : 0f;                       // is_online
        result[10] = terminal.card_present ? 1f : 0f;                    // card_present
        result[11] = isUnknown ? 1f : 0f;                                // unknown_merchant
        result[12] = mccRiskValue;                                        // mcc_risk
        result[13] = Clamp(merchant.avg_amount / 10000f);
    }
}
