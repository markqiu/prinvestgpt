meilisearch --master-key w1WgJu0hrDMEdhwXvecIbzBWmLXOmvgYP_TFAu7Vjv4
curl -X PATCH -H 'Content-Type: application/json' http://localhost:7700/experimental-features -d '{"vectorStore": true}'