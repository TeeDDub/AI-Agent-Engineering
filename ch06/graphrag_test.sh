pip install graphrag
mkdir -p ./ragtest/input
curl -L http://www.gutenberg.org/cache/epub/103/pg103.txt -o ./ragtest/input/book.txt


graphrag init --root ./ragtest

# ragtest 디렉터리 .env 파일에 GRAPHRAG_API_KEY 추가
set -a
[ -f .env ] && . ./.env
set +a
echo "GRAPHRAG_API_KEY=${OPENAI_API_KEY}" > ./ragtest/.env


graphrag index --root ./ragtest



graphrag query \
--root ./ragtest \
--method global \
"What are the key themes in this novel?"

graphrag query \
--root ./ragtest \
--method local \
"Who is Phileas Fogg and what motivates his journey?"