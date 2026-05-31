FROM mcr.microsoft.com/dotnet/sdk:10.0-alpine AS build
WORKDIR /src
COPY src/RinhaBackend2026FraudeApi/RinhaBackend2026FraudeApi.csproj .
RUN dotnet restore
COPY src/RinhaBackend2026FraudeApi/ .
RUN dotnet publish -c Release -o /app/publish

FROM mcr.microsoft.com/dotnet/aspnet:10.0-alpine
WORKDIR /app
COPY --from=build /app/publish .
COPY resources/ ./resources/
ENV RESOURCES_PATH=/app/resources
ENTRYPOINT ["dotnet", "RinhaBackend2026FraudeApi.dll"]
