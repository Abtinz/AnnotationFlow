FROM node:20-slim

WORKDIR /app

COPY package.json /app/package.json
RUN npm install

COPY index.html /app/index.html
COPY src /app/src

EXPOSE 8081

CMD ["npm", "run", "start"]

