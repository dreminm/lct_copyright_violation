FROM node:17-alpine as build
WORKDIR /front_app
COPY . /front_app
RUN npm install
RUN npm i vite
RUN npm run build
FROM nginx:1.16.0-alpine
COPY --from=build /front_app/dist /usr/share/nginx/html
EXPOSE 8001
CMD ["nginx", "-g", "daemon off;"]
