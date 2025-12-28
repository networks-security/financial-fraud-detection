import { Router } from "express";
import { getTransactions } from "./analyzed-transaction.controller.ts";

const dashboardRoutes = Router();
dashboardRoutes.get("/transactions", getTransactions);

export default dashboardRoutes;
