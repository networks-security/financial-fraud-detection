import express from "express";
import { errorHandler } from "./core/middleware/errorHandler.ts";

const app = express();

app.use(express.json());

// TODO: Routes

// Global error handling middleware (should be after routes)
app.use(errorHandler);

export default app;
