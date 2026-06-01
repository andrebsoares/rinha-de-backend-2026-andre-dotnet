using System.Text.Json.Serialization;

[JsonSerializable(typeof(FraudScoreRequest))]
[JsonSerializable(typeof(FraudScoreResponse))]
[JsonSourceGenerationOptions(PropertyNameCaseInsensitive = true)]
internal partial class AppJsonContext : JsonSerializerContext { }
