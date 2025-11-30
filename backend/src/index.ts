import express, { type Express } from "express";
// TODO: switch to HTTPS server later
import http from "http";
import { initializeSocketIO } from "./core/ws-server.js";
import dashboardRoutes from "./features/dashboard/dashboard.routes.ts";
import transactionsRoutes from "./features/transactions/transactions.routes.ts";
import cors from "cors";
import { spawn } from 'child_process';

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

const uploadScript = 'src/firestoreUploadScript.py';
const downloadScript = 'src/firestoreDownloadScript.py';
const args = ['src/test.json']; 

const pythonProcess = spawn('python', [uploadScript, ...args]);
const pythonProcess2 = spawn('python', [downloadScript, ...args]);

pythonProcess.stdout.on('data', (data) => {
    console.log(`Python stdout: ${data}`);
});

pythonProcess.stderr.on('data', (data) => {
    console.error(`Python stderr: ${data}`);
});

pythonProcess.on('close', (code) => {
    console.log(`Python process exited with code ${code}`);
});

pythonProcess2.stdout.on('data', (data) => {
    console.log(`Python stdout: ${data}`);
});

pythonProcess2.stderr.on('data', (data) => {
    console.error(`Python stderr: ${data}`);
});

pythonProcess2.on('close', (code) => {
    console.log(`Python process exited with code ${code}`);
});
