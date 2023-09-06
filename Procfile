web: gunicorn app:app --log-file=- -w 1 --max-requests 1000 --max-requests-jitter 50 --graceful-timeout 2 --timeout 10 --keep-alive 1
web: node --optimize_for_size --max_old_space_size=460 server.js