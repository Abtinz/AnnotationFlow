FROM node:20-slim

WORKDIR /app

COPY package.json /app/package.json
RUN npm install

COPY App.tsx /app/App.tsx

EXPOSE 8081

CMD ["npm", "run", "start"]

