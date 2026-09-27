# Build context = ./app/frontend
FROM nginx:1.27-alpine
COPY index.html /usr/share/nginx/html/index.html
COPY nginx.conf.template /etc/nginx/templates/default.conf.template
# BACKEND_URL doc tu bien moi truong luc container khoi dong (nginx envsubst templating)
ENV BACKEND_URL=http://backend:8000
EXPOSE 80
