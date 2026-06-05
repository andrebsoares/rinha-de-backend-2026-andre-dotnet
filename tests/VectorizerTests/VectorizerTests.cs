using Xunit;

public class VectorizerTests
{
    private static readonly IReadOnlyDictionary<string, float> MccRisk = new Dictionary<string, float>
    {
        ["5411"] = 0.15f,
        ["5912"] = 0.20f,
        ["7801"] = 0.80f,
    };

    // Payload 1: last_transaction = null → sentinela -1 nos índices 5 e 6
    [Fact]
    public void LastTransactionNull_Indices5And6_AreSentinel()
    {
        var req = MakeRequest(
            amount: 41.12f, installments: 2,
            requestedAt: new DateTime(2026, 3, 11, 18, 45, 53, DateTimeKind.Utc),
            avgAmount: 82.24f, txCount24h: 3,
            knownMerchants: ["MERC-003", "MERC-016"],
            merchantId: "MERC-016", mcc: "5411", merchantAvg: 60.25f,
            isOnline: false, cardPresent: true, kmFromHome: 29.23f,
            lastTx: null);

        Span<float> v = stackalloc float[14];
        Vectorizer.Vectorize(req, MccRisk, v);

        Assert.Equal(-1f, v[5]);
        Assert.Equal(-1f, v[6]);
    }

    // Índice 4: day_of_week — quarta-feira (2026-03-11) = 2 → 2/6 ≈ 0.3333
    // Armadilha: .NET DayOfWeek tem domingo=0, spec quer segunda=0
    [Fact]
    public void DayOfWeek_Wednesday_IsCorrectlyMapped()
    {
        // 2026-03-11 é quarta-feira → índice 2 na spec (seg=0) → 2/6
        var req = MakeRequest(
            requestedAt: new DateTime(2026, 3, 11, 18, 45, 53, DateTimeKind.Utc));

        Span<float> v = stackalloc float[14];
        Vectorizer.Vectorize(req, MccRisk, v);

        Assert.Equal(2f / 6f, v[4], precision: 4);
    }

    // Índice 5: minutes_since_last_tx — 20:23:35 - 14:58:35 = 325 min → 325/1440 ≈ 0.2257
    [Fact]
    public void MinutesSinceLastTx_CorrectTotalMinutes()
    {
        var req = MakeRequest(
            requestedAt: new DateTime(2026, 3, 11, 20, 23, 35, DateTimeKind.Utc),
            lastTx: new LastTransactionInfo(
                timestamp: new DateTime(2026, 3, 11, 14, 58, 35, DateTimeKind.Utc),
                km_from_current: 18.86f));

        Span<float> v = stackalloc float[14];
        Vectorizer.Vectorize(req, MccRisk, v);

        Assert.Equal(325f / 1440f, v[5], precision: 4);
    }

    // Índice 2: amount_vs_avg com clamp — 4368.82 / 68.88 / 10 = 6.34 → deve ser 1.0
    [Fact]
    public void AmountVsAvg_ExceedsOne_IsClamped()
    {
        var req = MakeRequest(amount: 4368.82f, avgAmount: 68.88f);

        Span<float> v = stackalloc float[14];
        Vectorizer.Vectorize(req, MccRisk, v);

        Assert.Equal(1.0f, v[2]);
    }

    // Índice 11: unknown_merchant — presente=0, ausente=1
    // Índice 12: mcc ausente no dict → default 0.5
    [Fact]
    public void UnknownMerchant_And_MccDefault()
    {
        // Merchant ausente da lista
        var reqUnknown = MakeRequest(merchantId: "MERC-NOVO", knownMerchants: ["MERC-001"], mcc: "9999");
        Span<float> v1 = stackalloc float[14];
        Vectorizer.Vectorize(reqUnknown, MccRisk, v1);
        Assert.Equal(1f, v1[11]); // desconhecido
        Assert.Equal(0.5f, v1[12]); // mcc ausente → default

        // Merchant conhecido
        var reqKnown = MakeRequest(merchantId: "MERC-001", knownMerchants: ["MERC-001"], mcc: "5411");
        Span<float> v2 = stackalloc float[14];
        Vectorizer.Vectorize(reqKnown, MccRisk, v2);
        Assert.Equal(0f, v2[11]); // conhecido
        Assert.Equal(0.15f, v2[12]); // mcc 5411
    }

    // --- helper ---

    private static FraudScoreRequest MakeRequest(
        float amount = 100f,
        int installments = 1,
        DateTime? requestedAt = null,
        float avgAmount = 100f,
        int txCount24h = 1,
        List<string>? knownMerchants = null,
        string merchantId = "MERC-001",
        string mcc = "5411",
        float merchantAvg = 100f,
        bool isOnline = false,
        bool cardPresent = true,
        float kmFromHome = 1f,
        LastTransactionInfo? lastTx = null)
    {
        return new FraudScoreRequest(
            id: "tx-test",
            transaction: new TransactionInfo(amount, installments,
                requestedAt ?? new DateTime(2026, 3, 11, 12, 0, 0, DateTimeKind.Utc)),
            customer: new CustomerInfo(avgAmount, txCount24h,
                knownMerchants ?? [merchantId]),
            merchant: new MerchantInfo(merchantId, mcc, merchantAvg),
            terminal: new TerminalInfo(isOnline, cardPresent, kmFromHome),
            last_transaction: lastTx);
    }
}
