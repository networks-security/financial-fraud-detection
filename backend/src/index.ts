import express, { type Express } from "express";
// TODO: switch to HTTPS server later
import http from "http";
import { initializeSocketIO } from "./core/ws-server.js";
import dashboardRoutes from "./features/dashboard/dashboard.routes.ts";
import transactionsRoutes from "./features/transactions/transactions.routes.ts";
import cors from "cors";

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
