import express, { type Express } from "express";
// TODO: switch to HTTPS server later
import http from "http";
import { initializeSocketIO } from "../websockets/ws-server.ts";
import dashboardRoutes from "./features/dashboard/dashboard.routes.ts";
import transactionsRoutes from "../../core/new-transaction/transactions.routes.ts";
import cors from "cors";
import * as dotenv from "dotenv";
import { ddosMiddleware } from "../../application/api/middleware/anti-ddos.middleware.ts";

dotenv.config();

const app: Express = express();
const port = process.env.PORT || 4000;

// CORS configuration
app.use(
  cors({
    origin: "*",
    methods: ["GET", "POST"],
  })
);

// Middleware to parse JSON request body
app.use(express.json());

// Apply DDoS middleware globally
app.use(ddosMiddleware);

// Register API Routes
app.use("/", dashboardRoutes);
app.use("/transaction", transactionsRoutes);

// Setup HTTP server
const httpServer = http.createServer(app);
// Attach Socket.IO
initializeSocketIO(httpServer);
httpServer.listen(port, () => {
  console.log(`HTTP server running on http://localhost:${port}`);
  console.log(`Socket.IO running on ws://localhost:${port}`);
});
