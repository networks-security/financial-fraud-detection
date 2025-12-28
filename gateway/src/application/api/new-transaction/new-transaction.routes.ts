import { Router } from "express";
import { processNewTransaction } from "./new-transaction.controller.ts";

const transactionsRoutes = Router();
transactionsRoutes.post("/process", processNewTransaction);

export default transactionsRoutes;
