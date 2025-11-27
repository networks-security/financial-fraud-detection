import { Router } from "express";
import { processNewTransaction } from "./transactions.controller.ts";

const transactionsRoutes = Router();
transactionsRoutes.post("/process", processNewTransaction);

export default transactionsRoutes;
