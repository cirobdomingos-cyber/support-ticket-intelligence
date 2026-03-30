install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

run:
	uvicorn main:app --reload --host 0.0.0.0 --port 8000

health:
	curl http://127.0.0.1:8000/health

route:
	curl -X POST http://127.0.0.1:8000/route -H "Content-Type: application/json" \
	-d '{"description":"Customer reports battery overheating during startup"}'

search:
	curl -X POST http://127.0.0.1:8000/search -H "Content-Type: application/json" \
	-d '{"description":"engine makes loud knocking noise during acceleration","top_k":5}'
