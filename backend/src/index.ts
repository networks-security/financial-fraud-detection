import express, { type Express, type Request, type Response } from "express";
// TODO: switch to HTTPS server later
import http from "http";
import { initializeSocketIO } from "./core/ws-server.js";

const app: Express = express();
const port = process.env.PORT || 3000;

app.get("/", (_req: Request, res: Response) => {
  res.send("ML Fraud Detection Backend");
});

const httpServer = http.createServer(app);

// Attach Socket.IO
initializeSocketIO(httpServer);

httpServer.listen(port, () => {
  console.log(`HTTP server running on http://localhost:${port}`);
  console.log(`Socket.IO running on ws://localhost:${port}`);
});
